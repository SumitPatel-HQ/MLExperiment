#McCulloch Pitts Model.
class MCPNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        # Weighted sum
        net = sum(x * w for x, w in zip(inputs, self.weights))

        # Step activation
        return 1 if net >= self.threshold else 0


def print_gate_results(name, neuron, test_inputs):
    print(f"\n{name}")
    for inp in test_inputs:
        output = neuron.activate(inp)
        print(f"Input: {inp} -> Output: {output}")


test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# OR Gate: fires if at least one input is 1
or_neuron = MCPNeuron(weights=[1, 1], threshold=1)
print_gate_results("OR Gate", or_neuron, test_inputs)

# AND Gate: fires only if all inputs are 1
and_neuron = MCPNeuron(weights=[1, 1], threshold=2)
print_gate_results("AND Gate", and_neuron, test_inputs)

# NOT Gate: single input, fires if input is 0
not_neuron = MCPNeuron(weights=[-1], threshold=0)
print_gate_results("\nNOT Gate", not_neuron, [(0,), (1,)])

# NAND Gate: fires unless all inputs are 1
nand_neuron = MCPNeuron(weights=[-1, -1], threshold=-1)
print_gate_results("NAND Gate", nand_neuron, test_inputs)

# NOR Gate: fires only if all inputs are 0
nor_neuron = MCPNeuron(weights=[-1, -1], threshold=0)
print_gate_results("NOR Gate", nor_neuron, test_inputs)

# XOR Gate: requires a network (not a single MCP neuron)
# XOR = (x1 AND NOT x2) OR (NOT x1 AND x2)
print("\nXOR Gate (Network)")
not_x1 = MCPNeuron(weights=[-1], threshold=0)
not_x2 = MCPNeuron(weights=[-1], threshold=0)
and1 = MCPNeuron(weights=[1, 1], threshold=2)  # x1 AND NOT x2
and2 = MCPNeuron(weights=[1, 1], threshold=2)  # NOT x1 AND x2
or_final = MCPNeuron(weights=[1, 1], threshold=1)

for inp in test_inputs:
    n1 = not_x1.activate((inp[0],))
    n2 = not_x2.activate((inp[1],))
    a1 = and1.activate((inp[0], n2))
    a2 = and2.activate((n1, inp[1]))
    output = or_final.activate((a1, a2))
    print(f"Input: {inp} -> Output: {output}")
