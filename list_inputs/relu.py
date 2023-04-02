
import sys 
sys.path.append('..')
from convert import *

file = 'relu.p64'

with open(file) as f:
    source = f.read()
node = ast.parse(source, filename = file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
stack_len = 100
print("s", stack_len)
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
x = [-100, 3, 5, 5, 23, -2, 4, 5,3, -4]
inputs = [[len(x)], x, [0]*len(x)]
# input, os, oe = relu_process(x, inp_start)
input, os, oe = process(inputs, inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)