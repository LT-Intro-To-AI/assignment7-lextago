from neural import NeuralNet

sq_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
    ([.9], [.81])
]
sqn = NeuralNet(1, 20, 1)
# sqn.train(sq_training_data)

# print()
# print(sqn.test_with_expected(sq_training_data))
# print(sqn.evaluate([0.66]))
# print(sqn.evaluate([0.95]))

xor_data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[0])]

xorn = NeuralNet(2, 1, 1)

# xorn.train(xor_data)

# print(xorn.test_with_expected(xor_data))

voter_opinion = [
    ([0.9, 0.6, 0.8, 0.3, 0.1],[1]),
    ([0.8, 0.8, 0.4, 0.6, 0.4],[1]),
    ([0.7, 0.2, 0.4, 0.6, 0.3],[1]),
    ([0.5, 0.5, 0.8, 0.4, 0.8],[0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8],[0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6],[0])
]

von = NeuralNet(5, 10, 1)

# von.train(voter_opinion)

# print(von.test_with_expected(voter_opinion))


test_data = [
    ([1, 1, 1, .1, .1]),
    ([.5, .2, .1, .7, .7]),
    ([.8, .3, .3, .3, .8]),
    ([.8, .3, .3, .8, .3]),
    ([.9, .8, .8, .3, .6])
]

# print(von.evaluate(test_data[0]))
# print(von.evaluate(test_data[1]))
# print(von.evaluate(test_data[2]))
# print(von.evaluate(test_data[3]))
# print(von.evaluate(test_data[4]))