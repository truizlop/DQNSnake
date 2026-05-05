import Foundation
import MLX
import SnakeEnv

struct DQNTrainer {
    let env: SnakeEnv
    var config: DQNTrainingConfig
    var replayBuffer: ReplayBuffer
    let learner: DQNLearner
    var explorationPolicy: EpsilonGreedyPolicy
    let episodeRunner: DQNEpisodeRunner
    let evaluator: DQNEvaluator
    var checkpointManager: DQNCheckpointManager
    var lastTrainStepMetrics: DQNTrainStepMetrics?
    var tensorBoardPublisher: TensorBoardMetricsPublisher?
    var structuredLogger: DQNStructuredLogger?
    var stabilityTracker: DQNStabilityTracker
    var bestTrainingEpisodeScore: Float?

    init(env: SnakeEnv, config: DQNTrainingConfig = DQNTrainingConfig()) {
        self.env = env
        self.config = config
        self.replayBuffer = ReplayBuffer(
            capacity: config.replayBufferCapacity,
            samplingStrategy: config.replaySamplingStrategy,
            prioritizedAlpha: config.prioritizedReplayAlpha,
            prioritizedEpsilon: config.prioritizedReplayEpsilon,
            seed: config.seed
        )
        self.learner = DQNLearner(
            gamma: config.gamma,
            learningRate: config.learningRate,
            dqnAlgorithm: config.dqnAlgorithm
        )
        self.explorationPolicy = EpsilonGreedyPolicy(
            epsilonStart: config.epsilonStart,
            epsilonEnd: config.epsilonEnd,
            epsilonDecaySteps: config.epsilonDecaySteps,
            seed: config.seed.map { $0 &+ 1 }
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
        self.structuredLogger = nil
        self.stabilityTracker = DQNStabilityTracker()
        self.bestTrainingEpisodeScore = nil
    }

    mutating func run() async throws {
        defer {
            tensorBoardPublisher?.stop()
            structuredLogger?.stop()
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
        if config.enableStructuredLogs {
            let logger = DQNStructuredLogger(outputPath: config.structuredLogPath)
            try logger.start()
            self.structuredLogger = logger
        }
        structuredLogger?.log(
            event: "run_start",
            step: 0,
            fields: [
                "seed": config.seed.map(String.init) ?? "none",
                "replay_sampling_strategy": "\(config.replaySamplingStrategy)",
                "prioritized_replay_alpha": "\(config.prioritizedReplayAlpha)",
                "prioritized_replay_beta_start": "\(config.prioritizedReplayBetaStart)",
                "prioritized_replay_beta_anneal_steps": "\(config.prioritizedReplayBetaAnnealSteps)",
                "prioritized_replay_epsilon": "\(config.prioritizedReplayEpsilon)",
                "train_every": "\(config.trainEvery)",
                "batch_size": "\(config.batchSize)",
                "max_consecutive_skipped_updates": "\(config.maxConsecutiveSkippedUpdates)",
                "max_loss_for_update": "\(config.maxLossForUpdate)",
                "max_abs_q_value": "\(config.maxAbsQValue)",
                "max_gradient_l2_norm": "\(config.maxGradientL2Norm)",
                "dqn_algorithm": config.dqnAlgorithm.rawValue,
            ]
        )

        var globalStep = try maybeResumeFromCheckpoint()
        var episode = 0

        while globalStep < config.totalEnvironmentSteps {
            episode += 1
            let episodeStart = Date()
            let episodeResult = try await episodeRunner.runEpisode(
                globalStep: &globalStep,
                totalEnvironmentSteps: config.totalEnvironmentSteps,
                selectAction: { state, step in
                    self.selectAction(state: state, globalStep: step)
                },
                onTransition: { transition, step in
                    self.replayBuffer.append(transition)
                    try self.applyTrainingSchedule(globalStep: step)
                    self.maybeEmitResourceTelemetry(globalStep: step)
                }
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
                    "train/episode_apples": Float(episodeResult.applesEaten),
                    "train/replay_size": Float(replayBuffer.count),
                    "train/replay_fill_ratio": Float(replayBuffer.count) / Float(config.replayBufferCapacity),
                    "train/epsilon": explorationPolicy.epsilon(at: globalStep),
                    "train/episode_duration_s": Float(Date().timeIntervalSince(episodeStart)),
                    "train/action_up": Float(episodeResult.actionCounts[.up] ?? 0),
                    "train/action_left": Float(episodeResult.actionCounts[.left] ?? 0),
                    "train/action_down": Float(episodeResult.actionCounts[.down] ?? 0),
                    "train/action_right": Float(episodeResult.actionCounts[.right] ?? 0),
                ]
            )
            structuredLogger?.log(
                event: "episode_end",
                step: globalStep,
                fields: [
                    "episode": "\(episode)",
                    "episode_steps": "\(episodeResult.steps)",
                    "episode_reward": "\(episodeResult.totalReward)",
                    "episode_score": "\(episodeResult.finalScore)",
                    "episode_apples": "\(episodeResult.applesEaten)",
                    "episode_duration_s": "\(Date().timeIntervalSince(episodeStart))",
                    "epsilon": "\(explorationPolicy.epsilon(at: globalStep))",
                    "replay_size": "\(replayBuffer.count)",
                    "replay_fill_ratio": "\(Float(replayBuffer.count) / Float(config.replayBufferCapacity))",
                    "action_up": "\(episodeResult.actionCounts[.up] ?? 0)",
                    "action_left": "\(episodeResult.actionCounts[.left] ?? 0)",
                    "action_down": "\(episodeResult.actionCounts[.down] ?? 0)",
                    "action_right": "\(episodeResult.actionCounts[.right] ?? 0)",
                ]
            )
            try await maybeCaptureBestTrainingEpisodeGIF(
                episode: episode,
                episodeResult: episodeResult,
                globalStep: globalStep
            )

            if shouldEvaluate(episode: episode) {
                let evaluation = try await evaluator.evaluate(
                    episodes: config.evalEpisodes,
                    selectAction: { state in
                        learner.greedyAction(for: state)
                    }
                )
                print(
                    "eval episode=\(episode) episodes=\(evaluation.episodes) avgReward=\(evaluation.averageReward) avgScore=\(evaluation.averageScore) avgApples=\(evaluation.averageApples)"
                )
                structuredLogger?.log(
                    event: "evaluation",
                    step: globalStep,
                    fields: [
                        "episode": "\(episode)",
                        "eval_episodes": "\(evaluation.episodes)",
                        "eval_avg_reward": "\(evaluation.averageReward)",
                        "eval_avg_score": "\(evaluation.averageScore)",
                        "eval_avg_apples": "\(evaluation.averageApples)",
                    ]
                )
                let savedBest = try checkpointManager.maybeSaveBestCheckpoint(
                    saver: learner,
                    metricValue: evaluation.averageScore,
                    metricName: "eval_avg_score",
                    metadata: [
                        "global_step": "\(globalStep)",
                        "episode": "\(episode)",
                        "eval_episodes": "\(evaluation.episodes)",
                        "eval_avg_reward": "\(evaluation.averageReward)",
                        "eval_avg_score": "\(evaluation.averageScore)",
                        "eval_avg_apples": "\(evaluation.averageApples)",
                    ]
                )
                if savedBest {
                    structuredLogger?.log(
                        event: "checkpoint_best_saved",
                        step: globalStep,
                        fields: [
                            "episode": "\(episode)",
                            "metric_name": "eval_avg_score",
                            "metric_value": "\(evaluation.averageScore)",
                        ]
                    )
                }
                tensorBoardPublisher?.publish(
                    step: globalStep,
                    scalars: [
                        "eval/avg_reward": evaluation.averageReward,
                        "eval/avg_score": evaluation.averageScore,
                        "eval/avg_apples": evaluation.averageApples,
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

    private mutating func selectAction(state: MLXArray, globalStep: Int) -> SnakeAction {
        explorationPolicy.selectAction(
            globalStep: globalStep,
            greedyAction: learner.greedyAction(for: state)
        )
    }

    private mutating func applyTrainingSchedule(globalStep: Int) throws {
        if shouldOptimize(globalStep: globalStep) {
            try optimizeFromReplay(globalStep: globalStep)
        }
        if shouldSyncTarget(globalStep: globalStep) {
            learner.syncTargetFromOnline()
            structuredLogger?.log(
                event: "target_sync",
                step: globalStep,
                fields: ["target_sync_every": "\(config.targetSyncEvery)"]
            )
        }
        let savedPeriodic = try checkpointManager.maybeSaveStepCheckpoint(saver: learner, globalStep: globalStep)
        if savedPeriodic {
            structuredLogger?.log(
                event: "checkpoint_periodic_saved",
                step: globalStep,
                fields: ["checkpoint_every_steps": "\(config.checkpointEverySteps)"]
            )
        }
    }

    private mutating func optimizeFromReplay(globalStep: Int) throws {
        let startedAt = Date()
        let sample = replayBuffer.sample(batchSize: config.batchSize, importanceSamplingBeta: prioritizedReplayBeta(at: globalStep))
        let batch = DQNBatch(transitions: sample.transitions, importanceWeights: sample.importanceWeights)
        let tdErrors = learner.tdErrors(batch: batch)
        replayBuffer.updatePriorities(indices: sample.indices, tdErrors: tdErrors)
        lastTrainStepMetrics = learner.trainStep(batch: batch)
        if let metrics = lastTrainStepMetrics {
            stabilityTracker.record(trainStepMetrics: metrics)
            try enforceStabilityGuards(metrics: metrics, globalStep: globalStep)
            let optimizeDuration = Float(Date().timeIntervalSince(startedAt))
            tensorBoardPublisher?.publish(
                step: globalStep,
                scalars: [
                    "train/loss": metrics.loss,
                    "train/q_mean_abs": metrics.meanAbsQ,
                    "train/q_max_abs": metrics.maxAbsQ,
                    "train/grad_l2": metrics.gradientL2Norm,
                    "train/skipped_update": metrics.skippedUpdate ? 1 : 0,
                    "train/consecutive_skipped_updates": Float(stabilityTracker.consecutiveSkippedUpdates),
                    "train/optimize_duration_s": optimizeDuration,
                ]
            )
            structuredLogger?.log(
                event: "optimize_step",
                step: globalStep,
                fields: [
                    "loss": "\(metrics.loss)",
                    "q_mean_abs": "\(metrics.meanAbsQ)",
                    "q_max_abs": "\(metrics.maxAbsQ)",
                    "grad_l2": "\(metrics.gradientL2Norm)",
                    "skipped_update": "\(metrics.skippedUpdate ? 1 : 0)",
                    "consecutive_skipped_updates": "\(stabilityTracker.consecutiveSkippedUpdates)",
                    "optimize_duration_s": "\(optimizeDuration)",
                    "batch_size": "\(config.batchSize)",
                    "replay_size": "\(replayBuffer.count)",
                    "replay_beta": "\(prioritizedReplayBeta(at: globalStep))",
                ]
            )
        }
    }

    private func prioritizedReplayBeta(at globalStep: Int) -> Float {
        guard config.replaySamplingStrategy == .prioritized else {
            return 1
        }
        guard config.prioritizedReplayBetaAnnealSteps > 0 else {
            return 1
        }
        let clamped = min(max(globalStep, 0), config.prioritizedReplayBetaAnnealSteps)
        let progress = Float(clamped) / Float(config.prioritizedReplayBetaAnnealSteps)
        let beta = config.prioritizedReplayBetaStart + (1 - config.prioritizedReplayBetaStart) * progress
        return min(max(beta, 0), 1)
    }

    private func maybeEmitResourceTelemetry(globalStep: Int) {
        guard config.resourceTelemetryEverySteps > 0 else {
            return
        }
        guard globalStep > 0, globalStep % config.resourceTelemetryEverySteps == 0 else {
            return
        }
        guard let telemetry = DQNProcessTelemetry.capture() else {
            return
        }

        tensorBoardPublisher?.publish(
            step: globalStep,
            scalars: [
                "runtime/rss_mb": Float(telemetry.rssBytes) / 1_048_576,
                "runtime/vmem_mb": Float(telemetry.virtualBytes) / 1_048_576,
                "runtime/cpu_user_s": Float(telemetry.userCPUSeconds),
                "runtime/cpu_system_s": Float(telemetry.systemCPUSeconds),
            ]
        )

        structuredLogger?.log(
            event: "resource_telemetry",
            step: globalStep,
            fields: [
                "rss_bytes": "\(telemetry.rssBytes)",
                "virtual_bytes": "\(telemetry.virtualBytes)",
                "cpu_user_seconds": "\(telemetry.userCPUSeconds)",
                "cpu_system_seconds": "\(telemetry.systemCPUSeconds)",
            ]
        )
    }

    private func enforceStabilityGuards(metrics: DQNTrainStepMetrics, globalStep: Int) throws {
        if metrics.loss.isFinite, metrics.loss > config.maxLossForUpdate {
            try throwAndLogGuardError(
                .lossExceeded(limit: config.maxLossForUpdate, observed: metrics.loss, step: globalStep),
                globalStep: globalStep
            )
        }
        if metrics.maxAbsQ.isFinite, metrics.maxAbsQ > config.maxAbsQValue {
            try throwAndLogGuardError(
                .qValueExceeded(limit: config.maxAbsQValue, observed: metrics.maxAbsQ, step: globalStep),
                globalStep: globalStep
            )
        }
        if metrics.gradientL2Norm.isFinite, metrics.gradientL2Norm > config.maxGradientL2Norm {
            try throwAndLogGuardError(
                .gradientNormExceeded(
                    limit: config.maxGradientL2Norm,
                    observed: metrics.gradientL2Norm,
                    step: globalStep
                ),
                globalStep: globalStep
            )
        }
        if stabilityTracker.consecutiveSkippedUpdates >= config.maxConsecutiveSkippedUpdates {
            try throwAndLogGuardError(
                .consecutiveSkippedUpdatesExceeded(
                    limit: config.maxConsecutiveSkippedUpdates,
                    observed: stabilityTracker.consecutiveSkippedUpdates,
                    step: globalStep
                ),
                globalStep: globalStep
            )
        }
    }

    private func throwAndLogGuardError(_ error: DQNTrainingGuardError, globalStep: Int) throws -> Never {
        tensorBoardPublisher?.publish(
            step: globalStep,
            scalars: [
                "train/stability_guard_triggered": 1,
            ]
        )
        structuredLogger?.log(
            event: "stability_guard_triggered",
            step: globalStep,
            fields: [
                "reason": error.localizedDescription,
            ]
        )
        throw error
    }

    private func episodeLogLine(episode: Int, episodeResult: DQNEpisodeResult, globalStep: Int) -> String {
        var line =
            "episode=\(episode) steps=\(episodeResult.steps) reward=\(episodeResult.totalReward) score=\(episodeResult.finalScore) apples=\(episodeResult.applesEaten) globalStep=\(globalStep)"
        if let metrics = lastTrainStepMetrics {
            line +=
                " trainLoss=\(metrics.loss) qMeanAbs=\(metrics.meanAbsQ) qMaxAbs=\(metrics.maxAbsQ) gradL2=\(metrics.gradientL2Norm) skippedUpdate=\(metrics.skippedUpdate)"
        }
        return line
    }

    private mutating func maybeCaptureBestTrainingEpisodeGIF(
        episode: Int,
        episodeResult: DQNEpisodeResult,
        globalStep: Int
    ) async throws {
        guard config.enableBestEpisodeGIFCapture else {
            return
        }
        let previousBest = bestTrainingEpisodeScore ?? -.infinity
        guard episodeResult.finalScore > previousBest else {
            return
        }

        bestTrainingEpisodeScore = episodeResult.finalScore

        let scoreTag = String(format: "%.3f", episodeResult.finalScore)
            .replacingOccurrences(of: "-", with: "m")
            .replacingOccurrences(of: ".", with: "p")
        let filename = "best_ep_\(episode)_step_\(globalStep)_score_\(scoreTag).gif"
        let directory = URL(fileURLWithPath: config.bestEpisodeGIFDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent(filename).path

        let frameCount = try await env.saveLastEpisodeGIF(
            path: path,
            scale: config.bestEpisodeGIFScale,
            frameDurationMs: config.bestEpisodeGIFFrameDurationMs
        )

        structuredLogger?.log(
            event: "best_episode_gif_saved",
            step: globalStep,
            fields: [
                "episode": "\(episode)",
                "episode_score": "\(episodeResult.finalScore)",
                "episode_apples": "\(episodeResult.applesEaten)",
                "gif_path": path,
                "gif_frames": "\(frameCount)",
            ]
        )
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
        structuredLogger?.log(
            event: "resume",
            step: resumedStep,
            fields: [
                "checkpoint_path": standardized,
                "resumed_step": "\(resumedStep)",
            ]
        )
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
