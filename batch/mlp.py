import sys 
sys.path.append('..')

from convert import *

file = 'mlp.p64'



def mlp_process(x, W1, W2,input_start):
    input_dim = x.shape[0]
    hidden_dim = W1.shape[0]
    output_dim = W2.shape[0]

    num_ins = 7
    out = [input_dim, hidden_dim, output_dim, input_start + num_ins]
    out.append(out[-1] + input_dim)
    out.append(out[-1] + input_dim*hidden_dim)
    out.append(out[-1] + hidden_dim*output_dim)
    
    out_start = out[-1]

    for i in range(input_dim):
        out += [x[i]]

    for i in range(hidden_dim):
        for j in range(input_dim):
            out += [W1[i,j]]

    for i in range(output_dim):
        for j in range(hidden_dim):
            out += [W2[i,j]]

    out += [0] * output_dim
    out_end = inp_start + len(out)
    return out, out_start, out_end

    

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
# print(inp_start)

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
print(x)
print(W1)
print(W2)
x1 = np.matmul(W1,x)
x1 = np.maximum(x1,0)
x1 = np.matmul(W2, x1)
print(list(x1))
inputs = [[input_dim], [hidden_dim], [output_dim], list(x), list(W1.flatten()), list(W2.flatten()), [0]*output_dim, [0]*output_dim, [0]*output_dim]
# input, os, oe = mlp_process(x,W1, W2, inp_start)
input, os, oe = process(inputs, inp_start)
print("OG", input)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)