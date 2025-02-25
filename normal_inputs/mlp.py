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


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
x = np.random.randint(low=-10, high=10,size=4)
W1 = np.random.randint(low=-10, high=10, size=(5,4))
W2 = np.random.randint(low=-10, high=10, size =(10,5))
print(x)
print(W1)
print(W2)
x1 = np.matmul(W1,x)
x1 = np.maximum(x1,0)
x1 = np.matmul(W2, x1)
print(list(x1))

input, os, oe = mlp_process(x,W1, W2, inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)