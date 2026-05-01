//
//  SnakeAction.swift
//  SnakeEnv
//
//  Created by Tomás Ruiz-López on 4/20/26.
//

enum SnakeAction: Int, CaseIterable {
    // Must match gym-snake action contract in snake_env.py:
    // 0=left, 1=up, 2=right, 3=down
    case left = 0
    case up = 1
    case right = 2
    case down = 3
}
