
from convert import reg, cmd, cmd_rev


class state:
    def __init__(self, stack_len) -> None:
        self.reg = reg
        self.cmd = cmd
        self.cmd_rev = cmd_rev
        self.num_registers = len(self.reg) + 2
        self.state = [0] * (self.num_registers + stack_len)
        self.func_start = [self.cmd['mov'], 
                           self.reg['rip'], 
                           self.reg['ofst'],
                           self.reg['dum'],
                           self.reg['dum'],
                           0,
                           0 ]
    
    def add_main_func(self, program, num_inputs):
        self.state += [0] * num_inputs
        op_code, args = program[0]
        self.state.append(op_code)
        self.state += args

        self.state += self.func_start

        for i in range(1,len(program)):
            op_code, args = program[i]
            self.state.append(op_code)
            self.state += args

        