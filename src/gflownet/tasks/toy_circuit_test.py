from typing import List

gate_1 = ["A", "B", "C", "D", "E", "R", "P"]
gate_2 = ["F", "G", "H", "I", "J", "P", "Q"]
gate_3 = ["K", "L", "M", "N", "O", "Q", "R"]

def parents(circuit_str: str) -> List[str]:
    gates = list(circuit_str)
    n = len(gates)
    if n == 0:
        return []
    parents: List[str] = []
    parents.append(''.join(gates[:-1]))
    first = gates[-1]
    for i in range(1, n):
        second = gates[-1 - i]
        if (first in gate_1 and second in gate_1) or (first in gate_2 and second in gate_2) or (first in gate_3 and second in gate_3):
            continue
        parents.append(''.join(gates[:-(i+1)] + gates[-i:]))
        break
    for j in range(i+1, n):
        third = gates[-j]
        if (third in gate_1 and (first in gate_1 or second in gate_1)) or (third in gate_2 and (first in gate_2 or second in gate_2)) or (third in gate_3 and (first in gate_3 or second in gate_3)):
            continue
        parents.append(''.join(gates[:-(j)] + gates[-j+1:]))
        break
    return parents

if __name__ == "__main__":
    print(parents("R"))  # ['AF', 'AP', 'FP']