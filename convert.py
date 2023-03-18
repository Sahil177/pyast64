from pyast64 import Compiler, Assembler, LocalsVisitor
import argparse
import ast
import sys
import re
import numpy as np

ref_find = re.compile(r".(%...).")
ref_find2 = re.compile(r"(%...,%...,.)")

reg = {
    "rax": 2,
    "rbx": 3,
    "rcx": 4,
    "rdx": 5,
    "rsi": 6,
    "rdi": 7,
    "rsp": 8,
    "rbp": 9,
    "rip": 10,
    "nlf": 11,
    "ef": 12,
    "gf": 13,
    "e": 14,
    "ofst": 15,
    "dum": 16,
    'r1' : 17
}

registers = len(reg) + 2
stack_len = 8

cmd = {
    "mov": 0,
    "add": 1,
    "mul": 2,
    "cmp": 3,
    "jz": 4,
    "jnl": 5,
    "inc": 6,
    "sub": 7,
    "leaq": 8,
    "jmp": 9,
    "ret": 10,
}

cmd = {
    "mov": 0,
    "mov1": 1,
    "mov2": 2,
    "mov3": 3,
    "add": 4,
    "add1": 5,
    "add2": 6,
    "add3": 7,
    "mul": 8,
    "mul1": 9,
    "mul2": 10,
    "mul3": 11,
    "sub": 12,
    "sub1": 13,
    "sub2": 14,
    "sub3": 15,
    "cmpe": 16,
    "cmpe1": 17,
    "cmpe2": 18,
    "cmpe3": 19,
    "cmpnl": 20,
    "cmpnl1": 21,
    "cmpnl2": 22,
    "cmpnl3": 23,
    "cmpg": 24,
    "cmpg1": 25,
    "cmpg2": 26,
    "cmpg3": 27,
    "jmp": 28,
    "jmp1": 29,
    "jmp2": 30,
    "jmp3": 31,
    "cjmp": 32,
    "inc": 33,
    "leaq": 34
}

cmd_rev = {cmd[i]: i for i in cmd}


def com_type(args):
    if type(args[0]) != list and type(args[1]) != list:
        arg = [args[0], args[1], "dum", "dum", 0, 0]
        return "", arg
    elif type(args[0]) == list and type(args[1]) != list:
        register, offset = args[0]
        if type(offset) == int:
            offset = int(offset / 8)
        arg = [register, args[1], "dum", "dum", offset, 0]
        return "1", arg
    elif type(args[0]) != list and type(args[1]) == list:
        register, offset = args[1]
        if type(offset) == int:
            offset = int(offset / 8)
        arg = [args[0], register, "dum", "dum", 0, offset]
        return "2", arg
    else:
        reg1, off1 = args[0]
        reg2, off2 = args[1]
        if type(off1) == int:
            off1 = int(off1 / 8)
        if type(off2) == int:
            off2 = int(off2 / 8)
        arg = [reg1, reg2, "dum", "dum", off1, off2]
        return "3", arg


def convert(total):
    stack_len = 0
    max_stack_len = 0
    converted = []
    for i in range(len(total)):
        op_code, args = total[i]
        new_args = []
        for arg in args:
            if arg[0] == "$" or arg[0] == "%":
                new_args.append(arg[1:])
            elif ref_find.search(arg) is not None:
                offset = ref_find.search(arg).span()[0]
                if arg[:offset] != "":
                    new_args.append(
                        [arg[offset + 2 : offset + 5], int(arg[:offset])]
                    )
                else:
                    if ref_find2.search(arg) is not None:
                        new_args.append([arg[2:5], arg[7:10]])
                    else:
                        new_args.append([arg[offset + 2 : offset + 5], 0])
            else:
                new_args.append(arg)

        if op_code == "pushq":
            stack_len += 1
            max_stack_len = max(max_stack_len, stack_len)
            converted.append(["sub", [1, "rsp", "dum", "dum", 0, 0]])
            suff, args = com_type([new_args[0], ["rsp", 0]])
            converted.append(["mov" + suff, args])
        elif op_code == "popq":
            stack_len -= 1
            suff, args = com_type([["rsp", 0], new_args[0]])
            converted.append(["mov" + suff, args])
            converted.append(["add", [1, "rsp", "dum", "dum", 0, 0]])
        elif op_code == "movq":
            suff, args = com_type(new_args)
            converted.append(["mov" + suff, args])
        elif op_code == "imulq":
            suff, args = com_type([new_args[0], "rax"])
            converted.append(["mul" + suff, args])
        elif op_code == "addq":
            suff, args = com_type(new_args)
            converted.append(["add" + suff, args])
        elif op_code == "shlq":
            suff, args = com_type([2 ** int(new_args[0]), new_args[1]])
            converted.append(["mul" + suff, args])
        elif op_code == "subq":
            suff, args = com_type(new_args)
            converted.append(["sub" + suff, args])
        elif op_code == "cmpq":
            suff, args = com_type(new_args)
            converted.append(["cmpe" + suff, args.copy()])
            converted.append(["cmpnl" + suff, args.copy()])
            converted.append(["cmpg" + suff, args.copy()])
        elif op_code == "incq":
            suff, args = com_type([1, new_args[0]])
            converted.append(["add" + suff, args])
        elif op_code == "jmp":
            suff, args = com_type([new_args[0], "rip"])
            args[2] = "ofst"
            converted.append(["jmp" + suff, args])
        elif op_code == "jz":
            converted.append(
                ["cjmp", [new_args[0], "rip", "ef", "ofst", 0, 0]]
            )
        elif op_code == "jnl":
            converted.append(
                ["cjmp", [new_args[0], "rip", "nlf", "ofst", 0, 0]]
            )
        elif op_code == 'call':
            stack_len += 1
            max_stack_len = max(max_stack_len, stack_len)
            converted.append(["sub", [1, "rsp", "dum", "dum", 0, 0]])
            suff, args = com_type(['rip', ["rsp", 0]])
            converted.append(["mov" + suff, args])
            suff, args = com_type([new_args[0], "rip"])
            args[2] = "ofst"
            converted.append(["jmp" + suff, args])
        elif op_code == 'ret':
            stack_len -= 1
            suff, args = com_type([["rsp", 0], 'r1'])
            converted.append(["mov" + suff, args])
            converted.append(["add", [1, "rsp", "dum", "dum", 0, 0]])
            suff, args = com_type([7, 'r1'])
            converted.append(["add" + suff, args])            
            suff, args = com_type(['r1', 'rip'])
            converted.append(["mov" + suff, args])
        elif op_code == 'leaq':
            suff, args = com_type(new_args)
            args[0] = args[4]
            args[4] = 0
            converted.append(["add", args])
        else:
            if len(new_args) == 0:
                new_args = [0, 0, 0, 0, 0, 0]
            suff, args = com_type(new_args)
            converted.append([op_code + suff, args])

    # print("stack_len", max_stack_len)
    return converted, max_stack_len


def convert_for_state(program):

    # Identify function calls and replace with line nums
    funcs = {}
    for i in range(len(program)):
        op_code, args = program[i]
        if op_code not in cmd:
            if op_code not in funcs:
                funcs[op_code] = 7 * i
                program[i][0] = 7 * i
            else:
                program[i][0] = funcs[op_code]

    print(funcs)
    # Update function call arguments
    for i in range(len(program) - 1):
        op_code, args = program[i]
        # print(op_code, args)
        for j in range(len(args)):
            if args[j] in funcs:
                program[i][1][j] = funcs[args[j]]

    # print(*program, sep="\n")

    # Extract constants and place at the end of the program.
    const_pos = 7 * len(program) + 1
    program.append([-1, []])
    consts = {}

    for i in range(len(program) - 1):
        op_code, args = program[i]
        if op_code not in funcs:
            for j in range(len(args)):
                if (args[j] not in reg) and (args[j] not in funcs):
                    arg = int(args[j])
                    if arg not in consts:
                        consts[arg] = const_pos
                        const_pos += 1
                        program[-1][1].append(arg)

    # print(consts)
    # print(*program, sep="\n")

    for i in range(len(program) - 1):
        op_code, args = program[i]
        if op_code not in funcs:
            # print('before{}'.format(program[i]))
            for j in range(len(args)):
                if (args[j] not in reg) and (args[j] not in funcs):
                    arg = int(args[j])
                    # print(arg, op_code, i, j)
                    # print(program[i])
                    program[i][1][j] = consts[arg]
            # print('after{}'.format(program[i]))

    # print(*program, sep='\n')

    for i in range(len(program) - 1):
        op_code, args = program[i]
        # convert register to nums
        for j in range(len(args)):
            if args[j] in reg:
                program[i][1][j] = reg[args[j]]
        if op_code in cmd:
            program[i][0] = cmd[op_code]

    return program


def execute(program, stack_len, inputs=[]):
    # state = [0] * (registers + 8 * stack_len) + inputs
    state = [0] * (registers + stack_len) + inputs

    # state[registers + 8 * stack_len + 8] = registers + 8 * stack_len + 16
    # print(state[56 : 56 + 88])
    # print(len([0] * (registers + 8 * stack_len)))
    # print(state[103:])
    # print(state[144])
    # print(state[160])
    # print(state[176])
    # print("INP START")
    # print(registers + 8 * stack_len)
    for i in range(len(program)):
        op_code, args = program[i]
        state.append(op_code)
        state += args

    # start = registers + 8 * stack_len + len(inputs)
    # end = start + 7 * (len(program) - 2)

    start = registers + stack_len + len(inputs)
    end = start + 7 * (len(program) - 2)
    # end = start + 18*7
    # end = start + 74*7
    print('end cmd', state[end:end+7])
    

    # print("init", state[0])
    state[reg["rip"]] = start
    state[reg["ofst"]] = start
    # state[reg["rsp"]] = registers + 8 * stack_len - 1

    state[reg["rsp"]] = registers + stack_len - 1
    # print(start)
    # print(state[start+240])

    i = 0
    while state[reg["rip"]] != end:
        execute_command(state)
        # if state[reg["rip"]] == start + 405:
        #     hit_break = True
        # if hit_break:
        #     print(state[state[reg["rip"]] : state[reg["rip"]] + 7])

        # print(state[reg['rip']])
        # print(state[state[reg['rbp']] + 8])
        # print(state[reg["rip"]]/7-7, state[reg['nlf']])
        # print(state[reg["rax"]])
        # print(
        #     "rax",
        #     state[reg["rax"]],
        #     "rsp",
        #     state[reg["rsp"]],
        #     "rsp-",
        #     state[state[reg["rsp"]]],
        #     "rdx",
        #     state[reg["rdx"]],
        # )
        i += 1

    print(i)
    # print(state[state[reg["rip"]]])
    # print(state[state[reg["rbp"]] + 16])
    return state


def execute_command(state):
    op_code = state[state[reg["rip"]]]
    args = state[state[reg["rip"]] + 1 : state[reg["rip"]] + 7]
    # print(op_code, args)

    for i in range(len(args)):
        if args[i] > registers:
            args[i] += state[reg["ofst"]]

    addr1, addr2, addr3, addr4, off1_add, off2_add = args

    # addr5 = state[addr1] + 8 * state[off1_add]
    # addr6 = state[addr2] + 8 * state[off2_add]

    # print(addr2, off2_add)

    addr5 = state[addr1] + state[off1_add]
    addr6 = state[addr2] + state[off2_add]


    addrs = [addr1, addr2, addr3, addr4, addr5, addr6]
    # print(addrs)

    for i in range(len(addrs)):
        if addrs[i] > len(state) or addrs[i] < 0 or not float(addrs[i]).is_integer():
            addrs[i] = 0

    addr1, addr2, addr3, addr4, addr5, addr6 = addrs

    val1 = state[addr1]
    val2 = state[addr2]
    val3 = state[addr3]
    val4 = state[addr4]
    val5 = state[addr5]
    val6 = state[addr6]

    # print('addr')
    # print(addr1, addr2, addr3, addr4)

    # print('val')
    # print(val1, val2, val3, val4)

    out_val = comp_result(op_code, val1, val2, val3, val4, val5, val6)
    out_addr = comp_addr(op_code, addr1, addr2, addr3, addr4, addr5, addr6)
    # print('---')
    # print(val1, val2, val3, val4, val5, val6)
    # print(out_addr)
    # print(out_val)
    # print('---')

    # if op_code =='cmp':
    #     print(out_val, out_addr)
    # print(out_addr)
    if out_addr is not None and out_val is not None:
        state[out_addr] = out_val

    state[reg["rip"]] += 7


def comp_result(op_code, val1, val2, val3, val4, val5, val6):
    if op_code in cmd_rev:
        op_code = cmd_rev[op_code]
    if op_code == "mov" or op_code == "mov2":
        return val1
    elif op_code == "mov1" or op_code == "mov3":
        # print("here")
        # print("val5 ", val5)
        return val5
    elif op_code == "add":
        return val1 + val2
    elif op_code == "add1":
        return val5 + val2
    elif op_code == "add2":
        return val1 + val6
    elif op_code == "add3":
        return val5 + val6
    elif op_code == "sub":
        return val2 - val1
    elif op_code == "sub1":
        return val2 - val5
    elif op_code == "sub2":
        return val6 - val1
    elif op_code == "sub3":
        return val6 - val5
    elif op_code == "mul":
        return val1 * val2
    elif op_code == "mul1":
        return val5 * val2
    elif op_code == "mul2":
        return val1 * val6
    elif op_code == "mul3":
        return val5 * val6
    elif op_code == "cmpe":
        return int(val2 == val1)
    elif op_code == "cmpe1":
        return int(val2 == val5)
    elif op_code == "cmpe2":
        return int(val6 == val1)
    elif op_code == "cmpe3":
        return int(val6 == val5)
    elif op_code == "cmpnl":
        return int(val2 >= val1)
    elif op_code == "cmpnl1":
        return int(val2 >= val5)
    elif op_code == "cmpnl2":
        return int(val6 >= val1)
    elif op_code == "cmpnl3":
        return int(val6 >= val5)
    elif op_code == "cmpg":
        return int(val2 > val1)
    elif op_code == "cmpg1":
        return int(val2 > val5)
    elif op_code == "cmpg2":
        return int(val6 > val1)
    elif op_code == "cmpg3":
        return int(val6 > val5)
    elif op_code == "jmp" or op_code == "jmp2":
        return val1 + val3
    elif op_code == "jmp1" or op_code == "jmp3":
        return val5 + val3
    elif op_code == "cjmp":
        # print(val1, val2, val3)
        if val3:
            return val1 + val4
        else:
            return val2  # + val4
    else:
        return None


def comp_addr(op_code, addr1, addr2, addr3, addr4, addr5, addr6):
    if op_code in cmd_rev:
        op_code = cmd_rev[op_code]
    set1 = set(
        [
            "mov",
            "mov1",
            "add",
            "add1",
            "sub",
            "sub1",
            "mul",
            "mul1",
            "jmp",
            "jmp1",
        ]
    )
    set2 = set(
        [
            "mov2",
            "mov3",
            "add2",
            "add3",
            "sub2",
            "sub3",
            "mul2",
            "mul3",
            "jmp2",
            "jmp3",
        ]
    )
    set3 = set(["cmpe", "cmpe1", "cmpe2", "cmpe3"])
    set4 = set(["cmpg", "cmpg1", "cmpg2", "cmpg3"])
    set5 = set(["cmpnl", "cmpnl1", "cmpnl2", "cmpnl3"])
    if op_code in set1:
        return addr2
    elif op_code in set2:
        # print("addr6", addr6)
        return addr6
    elif op_code in set3:
        return reg["ef"]
    elif op_code in set4:
        return reg["gf"]
    elif op_code in set5:
        return reg["nlf"]
    elif op_code == "cjmp":
        return reg["rip"]
    else:
        # print(op_code)
        return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename to compile")
    parser.add_argument(
        "-n",
        "--no-peephole",
        action="store_true",
        help="enable peephole assembler optimizer",
    )
    args = parser.parse_args()

    with open(args.filename) as f:
        source = f.read()
    node = ast.parse(source, filename=args.filename)
    compiler = Compiler(peephole=not args.no_peephole)
    compiler.compile(node)
    program = compiler.asm.total
    # print(*program, sep="\n")
    program, stack_len = convert(program)
    # print("s", stack_len)
    stack_len = 10
    # inp_start = registers + 8 * stack_len
    inp_start = registers + stack_len
    # print(*program, sep="\n")
    program = convert_for_state(program)
    print(*program, sep="\n")
    for i in range(len(program)):
        print(i, program[i])
