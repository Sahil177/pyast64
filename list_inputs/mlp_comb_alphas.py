import sys 
sys.path.append('..')

from convert import *

file = 'skillcomb_alpha.p64'


with open(file) as f:
    source = f.read()
node = ast.parse(source, filename = file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
stack_len = 2000
print("s", stack_len)
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
input_dim = 4
hidden_dim = 5
output_dim = 10
x = np.random.randint(low=-10, high=10,size=input_dim)
W1 = np.random.randint(low=-10, high=10, size=(hidden_dim,input_dim))
W2 = np.random.randint(low=-10, high=10, size =(output_dim,hidden_dim))
Ws1 = np.eye(9, dtype=np.int64)
Ws1[2] = 0
Ws1[2,3] = 1
Ws1[3] = 0
Ws1[3,4] = 1
Ws1[4] = 0
Ws1[4,6] = 1

print(Ws1)

Ws2 = np.eye(9, dtype=np.int64)
Ws2[0] = 0
Ws2[0,1] = 1
Ws2[1] = 0
Ws2[1,6] = 1
Ws2[2] = 0
Ws2[2,7] = 1

print(Ws2)

Ws3 = np.eye(9, dtype=np.int64)
Ws3[0] = 0
Ws3[0,1] = 1
Ws3[1] = 0
Ws3[1,2] = 1
Ws3[2] = 0
Ws3[2,7] = 1
Ws3[3] = 0
Ws3[3,5] = 1
Ws3[4] = 0
Ws3[4,8] =1

print(Ws3)

print(x)
print(W1)
print(W2)
x1 = np.matmul(W1,x)
x1 = np.maximum(x1,0)
x1 = np.matmul(W2, x1)
print(list(x1))
inputs = [[9],
          [input_dim], 
          [hidden_dim], 
          [output_dim], 
          list(x), 
          list(W1.flatten()), 
          list(W2.flatten()), 
          [0]*output_dim, 
          [0]*output_dim, 
          [0]*output_dim,
          [3], # timesteps
          [2], # num skills
          list(Ws1.flatten()),
          list(Ws2.flatten()),
          list(Ws3.flatten()),
          [1,0],
          [0,1],
          [1,0]]

lens = [len(lst) for lst in inputs]

# input, os, oe = mlp_process(x,W1, W2, inp_start)
input, os, oe = process(inputs, inp_start)
# print("OG", input)
# lens = [len(lst) for lst in inputs]
os = inp_start + len(inputs) + 1 + sum(lens[:9])
oe = inp_start + len(inputs) + 1 + sum(lens[:10])
print(inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)