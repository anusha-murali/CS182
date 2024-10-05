dictionary = GhostDictionary("dict2.txt")
prefix = "wal"
play_game(dictionary, prefix, 0, MinimaxAgent, MinimaxAgent)

prefix = "acr"
play_game(dictionary, prefix, 0, AlphaBetaAgent, AlphaBetaAgent)

play_game(dictionary, prefix, 0, OptimizedAgainstRandomAgent, RandomAgent)

play_game(dictionary, prefix, 0, RandomAgent, OptimizedAgainstRandomAgent)



