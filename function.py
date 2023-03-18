
from convert import *

file = 'function_test.p64'



def relu_process(x, input_start):
    N = len(x)

    num_ins = 3
    out = [N, inp_start + num_ins,inp_start + num_ins + N ]

    out_start = out[-1]

    for i in range(N):
        out += [x[i]]

    out += [0] * N
    out_end = inp_start + len(out)
    return out, out_start, out_end

    

with open(file) as f:
    source = f.read()
node = ast.parse(source, filename = file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
# print("s", stack_len)
stack_len = 100
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
x = [-100, 3, 45, 60, -2]
input, os, oe = relu_process(x, inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print('x', x)
print('squared', out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)