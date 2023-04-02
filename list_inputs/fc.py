import sys 
sys.path.append('..')

from convert import *

file = 'fc-m.p64'

def fc_process(W, x, inp_start):
    N = len(x)
    M = len(W)

    num_ins = 5
    out = [N, M, inp_start + num_ins, inp_start + num_ins + N, inp_start + num_ins + N + N*M] # [N, M, x add, W add, out add]

    out_start = out[-1]

    for i in range(N):
        out += [x[i]]

    for i in range(M):
        for j in range(N):
            out += [W[i][j]]

    out += [0] * M
    out_end = inp_start + len(out)

    return out, out_start, out_end


def process_out(out):
    cond_out = []

    for i in range(0, len(out), 8):
        cond_out.append(out[i])

    return cond_out

def test(iters, max_N, max_M, inp_start, program, stack_len):
    correct = 0

    for i in range(iters):
        N = np.random.randint(1,max_N)
        M = np.random.randint(1,max_M)
        # x = np.random.randint(100, size=(N))
        # W = np.random.randint(100, size=(M,N))

        x = 100*np.random.random(size=(N))
        W = 100*np.random.random(size=(M,N))

        # print(W)
        # print(x)
        res = np.matmul(W, x)
        # print(res)
        # inp, os, oe = fc_process(W, x, inp_start)
        W_flat = list(W.flatten())
        inputs = [[N], [M], list(x), W_flat, [0]*M]
        # input, os, oe = fc_process(W, x, inp_start)
        inp, os, oe = process(inputs, inp_start)
        state = execute(program, stack_len, inp)
        # out = process_out(state[os:oe])
        out = state[os:oe]
        # print(out)
        error = np.linalg.norm(res- np.array(out))
        if error < 1e-6:
            correct +=1

    return correct/iters





with open(file) as f:
    source = f.read()
node = ast.parse(source, filename = file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
print("s", stack_len)
stack_len = 1000
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
# print(*program, sep="\n")
x = [6, 3, 5]
W = [[12, 13, 5], [4, 5, 5], [2, 2, 2], [5,6,3]]
W_flat = [item for sublist in W for item in sublist]
inputs = [[len(x)], [len(W)], x, W_flat, [0]*len(W)]
# input, os, oe = fc_process(W, x, inp_start)
input, os, oe = process(inputs, inp_start)
print(input)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print('x', x)
print('w', W)
print(out)
print(np.matmul(W,x))
acc = test(100, 10 ,10, inp_start, program, stack_len)
print(acc)