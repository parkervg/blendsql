from copy import deepcopy

from typing import List, Union, Any
from collections import OrderedDict


class Tree:
    data: str
    children: List[Union[str, "Tree"]]

    def __init__(self, data: str, children: List[Any]) -> None:
        self.data = data
        self.children = children

    def __repr__(self) -> str:
        return f"Tree({self.data}, {self.children})"

    def _pretty(self, level, indent_str):
        yield f"{indent_str*level}{self.data}"
        if len(self.children) == 1 and not isinstance(self.children[0], Tree):
            yield f"\t{self.children[0]}\n"
        else:
            yield "\n"
            for n in self.children:
                if isinstance(n, Tree):
                    yield from n._pretty(level + 1, indent_str)
                else:
                    yield f"{indent_str*(level+1)}{n}\n"

    def pretty(self, indent_str: str = "  ") -> str:
        return "".join(self._pretty(0, indent_str))

    def iter_subtrees(self):
        return self.iter_subtrees_dfs()

    def iter_subtrees_dfs(self):
        queue = [self]
        subtrees = OrderedDict()
        for subtree in queue:
            subtrees[id(subtree)] = subtree
            queue += [
                c
                for c in reversed(subtree.children)
                if isinstance(c, Tree) and id(c) not in subtrees
            ]

        del queue
        return reversed(list(subtrees.values()))

    def iter_subtrees_bfs(self):
        stack = [self]
        stack_append = stack.append
        stack_pop = stack.pop
        while stack:
            node = stack_pop()
            if not isinstance(node, Tree):
                continue
            yield node
            for child in reversed(node.children):
                stack_append(child)

    def find_pred(self, pred):
        return filter(pred, self.iter_subtrees())

    def find_data(self, data: str):
        return self.find_pred(lambda t: t.data == data)

    def __deepcopy__(self, memo):
        return type(self)(self.data, deepcopy(self.children, memo))

    def copy(self):
        return type(self)(self.data, self.children)

    def set(self, data, children) -> None:
        self.data = data
        self.children = children
