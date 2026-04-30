import Foundation
import MLX
import SnakeEnv

struct DQNTrainer {
    let env: SnakeEnv
    var config: DQNTrainingConfig
    var replayBuffer: ReplayBuffer
    let learner: DQNLearner
    let explorationPolicy: EpsilonGreedyPolicy
    let episodeRunner: DQNEpisodeRunner
    let evaluator: DQNEvaluator
    var checkpointManager: DQNCheckpointManager
    var lastTrainStepMetrics: DQNTrainStepMetrics?
    var tensorBoardPublisher: TensorBoardMetricsPublisher?

    init(env: SnakeEnv, config: DQNTrainingConfig = DQNTrainingConfig()) {
        self.env = env
        self.config = config
        self.replayBuffer = ReplayBuffer(
            capacity: config.replayBufferCapacity,
            samplingStrategy: config.replaySamplingStrategy
        )
        self.learner = DQNLearner(gamma: config.gamma, learningRate: config.learningRate)
        self.explorationPolicy = EpsilonGreedyPolicy(
            epsilonStart: config.epsilonStart,
            epsilonEnd: config.epsilonEnd,
            epsilonDecaySteps: config.epsilonDecaySteps
        )
        self.episodeRunner = DQNEpisodeRunner(
            env: env,
            maxStepsPerEpisode: config.maxStepsPerEpisode
        )
        self.evaluator = DQNEvaluator(
            env: env,
            maxStepsPerEpisode: config.maxStepsPerEpisode
        )
        self.checkpointManager = DQNCheckpointManager(
            checkpointDirectory: config.checkpointDirectory,
            checkpointEverySteps: config.checkpointEverySteps,
            saveBestCheckpoint: config.saveBestCheckpoint
        )
        self.lastTrainStepMetrics = nil
        self.tensorBoardPublisher = nil
    }

    mutating func run() async throws {
        defer {
            tensorBoardPublisher?.stop()
        }

        if config.enableTensorBoard {
            let publisher = TensorBoardMetricsPublisher(
                logDir: config.tensorBoardLogDir,
                launchTensorBoard: config.launchTensorBoard,
                tensorBoardPort: config.tensorBoardPort,
                pythonExecutable: Self.pythonExecutable()
            )
            try publisher.start()
            self.tensorBoardPublisher = publisher
        }

        var globalStep = try maybeResumeFromCheckpoint()
        var episode = 0

        while globalStep < config.totalEnvironmentSteps {
            episode += 1
            let episodeResult = try await episodeRunner.runEpisode(
                globalStep: &globalStep,
                totalEnvironmentSteps: config.totalEnvironmentSteps,
                selectAction: { state, step in
                    self.selectAction(state: state, globalStep: step)
                },
                onTransition: { transition, step in
                    self.replayBuffer.append(transition)
                    try self.applyTrainingSchedule(globalStep: step)
                }
            )

            try checkpointManager.maybeSaveBestCheckpoint(
                saver: learner,
                episodeResult: episodeResult,
                episode: episode,
                globalStep: globalStep
            )

            print(episodeLogLine(
                episode: episode,
                episodeResult: episodeResult,
                globalStep: globalStep
            ))
            tensorBoardPublisher?.publish(
                step: globalStep,
                scalars: [
                    "train/episode_reward": episodeResult.totalReward,
                    "train/episode_score": episodeResult.finalScore,
                    "train/episode_steps": Float(episodeResult.steps),
                ]
            )

            if shouldEvaluate(episode: episode) {
                let evaluation = try await evaluator.evaluate(
                    episodes: config.evalEpisodes,
                    selectAction: { state in
                        learner.greedyAction(for: state)
                    }
                )
                print(
                    "eval episode=\(episode) episodes=\(evaluation.episodes) avgReward=\(evaluation.averageReward) avgScore=\(evaluation.averageScore)"
                )
                tensorBoardPublisher?.publish(
                    step: globalStep,
                    scalars: [
                        "eval/avg_reward": evaluation.averageReward,
                        "eval/avg_score": evaluation.averageScore,
                    ]
                )
            }
        }
    }

    private func shouldOptimize(globalStep: Int) -> Bool {
        globalStep >= config.warmupSteps
            && globalStep % config.trainEvery == 0
            && replayBuffer.canSample(batchSize: config.batchSize)
    }

    private func shouldSyncTarget(globalStep: Int) -> Bool {
        config.targetSyncEvery > 0
            && globalStep > 0
            && globalStep % config.targetSyncEvery == 0
    }

    private func shouldEvaluate(episode: Int) -> Bool {
        config.evalEveryEpisodes > 0
            && config.evalEpisodes > 0
            && episode % config.evalEveryEpisodes == 0
    }

    private mutating func applyTrainingSchedule(globalStep: Int) throws {
        if shouldOptimize(globalStep: globalStep) {
            optimizeFromReplay(globalStep: globalStep)
        }
        if shouldSyncTarget(globalStep: globalStep) {
            learner.syncTargetFromOnline()
        }
        try checkpointManager.maybeSaveStepCheckpoint(saver: learner, globalStep: globalStep)
    }

    private func selectAction(state: MLXArray, globalStep: Int) -> SnakeAction {
        explorationPolicy.selectAction(
            globalStep: globalStep,
            greedyAction: learner.greedyAction(for: state)
        )
    }

    private mutating func optimizeFromReplay(globalStep: Int) {
        let transitions = replayBuffer.sample(batchSize: config.batchSize)
        let batch = DQNBatch(transitions: transitions)
        lastTrainStepMetrics = learner.trainStep(batch: batch)
        if let metrics = lastTrainStepMetrics {
            tensorBoardPublisher?.publish(
                step: globalStep,
                scalars: [
                    "train/loss": metrics.loss,
                    "train/q_mean_abs": metrics.meanAbsQ,
                    "train/q_max_abs": metrics.maxAbsQ,
                    "train/grad_l2": metrics.gradientL2Norm,
                    "train/skipped_update": metrics.skippedUpdate ? 1 : 0,
                ]
            )
        }
    }

    private func episodeLogLine(episode: Int, episodeResult: DQNEpisodeResult, globalStep: Int) -> String {
        var line =
            "episode=\(episode) steps=\(episodeResult.steps) reward=\(episodeResult.totalReward) score=\(episodeResult.finalScore) globalStep=\(globalStep)"
        if let metrics = lastTrainStepMetrics {
            line +=
                " trainLoss=\(metrics.loss) qMeanAbs=\(metrics.meanAbsQ) qMaxAbs=\(metrics.maxAbsQ) gradL2=\(metrics.gradientL2Norm) skippedUpdate=\(metrics.skippedUpdate)"
        }
        return line
    }

    private func maybeResumeFromCheckpoint() throws -> Int {
        guard let resumePath = config.resumeCheckpointPath, !resumePath.isEmpty else {
            return 0
        }

        let standardized = (resumePath as NSString).standardizingPath
        let url = URL(fileURLWithPath: standardized)
        let metadata = try learner.loadOnlineModel(from: url)
        let resumedStep = Int(metadata["global_step"] ?? "") ?? 0
        print("resumed from checkpoint=\(standardized) globalStep=\(resumedStep)")
        return resumedStep
    }

    private static func pythonExecutable() -> String {
        if let configured = ProcessInfo.processInfo.environment["SNAKE_PYTHON_EXE"], !configured.isEmpty {
            return configured
        }
        if FileManager.default.fileExists(atPath: "/opt/anaconda3/bin/python3") {
            return "/opt/anaconda3/bin/python3"
        }
        return "/usr/bin/python3"
    }
}
