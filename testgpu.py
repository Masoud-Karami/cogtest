all_experiments = {'ProbabilisticReasoning', 'HorizonTask', 'RestlessBandit',
                       'InstrumentalLearning', 'TwoStepTask', 'BART', 'SerialMemoryTask', 'TemporalDiscounting'}

excluded_experiments = {'ProbabilisticReasoning', 'HorizonTask', 'RestlessBandit', 'InstrumentalLearning',
                            'TwoStepTask', 'BART', 'SerialMemoryTask', 'TemporalDiscounting'}

print(f'focusing folder: {list((all_experiments - excluded_experiments).elements())}')