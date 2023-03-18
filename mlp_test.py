
from convert import *

file = 'mlp_test.p64'



def mlp_process(x, W1 ,input_start):
    input_dim = x.shape[0]
    hidden_dim = W1.shape[0]

    num_ins = 5
    out = [input_dim, hidden_dim, input_start + num_ins]
    out.append(out[-1] + input_dim)
    out.append(out[-1] + input_dim*hidden_dim)
    
    out_add = inp_start + len(out) -1
    print(inp_start)
    out_start = out[-1]

    for i in range(input_dim):
        out += [x[i]]

    for i in range(hidden_dim):
        for j in range(input_dim):
            out += [W1[i,j]]


    out += [0] * hidden_dim
    out_end = inp_start + len(out)
    return out, out_start, out_end, out_add, hidden_dim

    

with open(file) as f:
    source = f.read()
node = ast.parse(source, filename = file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
stack_len = 1000
print("s", stack_len)
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
x = np.random.randint(10,size=20)
W1 = np.random.randint(10, size=(5,20))
print(x)
print(W1)
print(np.matmul(W1,x))
input, os, oe, out_add, out_dim = mlp_process(x,W1, inp_start)
print(input)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
out2 = state[state[out_add]: state[out_add] + out_dim]
print(out2)


print(state[inp_start-30:inp_start+6])
print(inp_start-9)
print(state[out_add])
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)