import sys 
sys.path.append('..')

from convert import *

file = 'W-test.p64'


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
Ws1 = np.eye(9, dtype=np.int64)
Ws1[2] = 0
Ws1[2,3] = 1
Ws1[3] = 0
Ws1[3,4] = 1
Ws1[4] = 0
Ws1[4,6] = 1

print(Ws1)

# x = np.random.randint(-10,10,9)
x = range(9)


inputs = [[9], 
          list(x), 
          list(Ws1.flatten())
]



lens = [len(lst) for lst in inputs]

# input, os, oe = mlp_process(x,W1, W2, inp_start)
input, os, oe = process(inputs, inp_start)
print("OG", input)
lens = [len(lst) for lst in inputs]
print(inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
print(state[2023:2032])
print(state[inp_start-50:inp_start])
print(np.matmul(Ws1,x))
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)