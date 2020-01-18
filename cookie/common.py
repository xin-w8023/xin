

def add_indent(s_: str='', num_space: int=2) -> str:
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_space * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s