import sys

sys.path.append("..")

from convert import *

from weights import *

file = "mlp.p64"


def generate_dataset_quad(num_points, a, b, c):
    inputs = []
    labels = [0] * num_points
    for i in range(num_points):
        input1 = 20 * (np.random.random() - 0.5)
        input2 = 200 * (np.random.random() - 0.5)
        inputs.append(input1)
        inputs.append(input2)
        if a * input1**2 + b * input1 + c > input2:
            labels[i] = 1

    return inputs, labels


with open(file) as f:
    source = f.read()
node = ast.parse(source, filename=file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
stack_len = 5000
print("s", stack_len)
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len
# print(inp_start)

print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
num_points = 100
input_dim = 2
hidden_dim = 10
output_dim = 1
# x = np.random.randint(low=-10, high=10, size=(num_points, input_dim))
# W1 = np.random.randint(low=-10, high=10, size=(hidden_dim, input_dim))
# b1 = np.random.randint(low=-10, high=10, size=(hidden_dim))
# W2 = np.random.randint(low=-10, high=10, size=(hidden_dim, hidden_dim))
# b2 = np.random.randint(low=-10, high=10, size=(hidden_dim))
# W3 = np.random.randint(low=-10, high=10, size=(output_dim, hidden_dim))
# b3 = np.random.randint(low=-10, high=10, size=(output_dim))

# print(x)
# print(W1)
# print(b1)
# print(W2)
# print(b2)
# print(W3)
# print(b3)

a = 1
b = 1
c = -20

test_inps, test_labels = generate_dataset_quad(num_points, a, b, c)

W1_ = []

for i in range(len(W1)):
    for j in range(len(W1[0])):
        W1_.append(W1[i][j])

W2_ = []

for i in range(len(W2)):
    for j in range(len(W2[0])):
        W2_.append(W2[i][j])

W3_ = []

for i in range(len(W3)):
    for j in range(len(W3[0])):
        W3_.append(W3[i][j])

sum = 0
# for i in range(num_points):
#     x1 = x[i]
#     x1 = np.matmul(W1, x1) + b1
#     x1 = np.maximum(x1, 0)
#     x1 = np.matmul(W2, x1) + b2
#     x1 = np.maximum(x1, 0)
#     x1 = np.matmul(W3, x1) + b3
#     print(i, x1)
#     sum += x1[0]

# print(sum)
# print(x.flatten())
# inputs = [
#     [num_points],
#     list(x.flatten()),
#     [1, 0],
#     [input_dim],
#     [hidden_dim],
#     [output_dim],
#     list(W1.flatten()),
#     list(b1),
#     list(W2.flatten()),
#     list(b2),
#     list(W3.flatten()),
#     list(b3),
#     [0],
# ]

inputs = [
    [num_points],
    test_inps,
    test_labels,
    [input_dim],
    [hidden_dim],
    [output_dim],
    W1_,
    b1,
    W2_,
    b2,
    W3_,
    b3[0],
    [0],
]

# inputs = [
#     [num_points],
#     [input_dim],
#     [hidden_dim],
#     [output_dim],
#     list(x.flatten()),
#     list(W1.flatten()),
#     list(b1),
#     list(W2.flatten()),
#     list(b2),
#     list(W3.flatten()),
#     list(b3),
#     [0] * output_dim * num_points,
# ]
# input, os, oe = mlp_process(x,W1, W2, inp_start)
input, os, oe = process(inputs, inp_start)
# print("OG", input)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# print(x1)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)
