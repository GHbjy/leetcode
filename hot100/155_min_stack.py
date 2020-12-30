class MinStack_origin:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self._list = []

    def push(self, x: int) -> None:
        self._list.append(x)
        return None

    def pop(self) -> None:
        self._list.pop()
        return None

    def top(self) -> int:
        return self._list[-1]

    def getMin(self) -> int:
        # _min = [x for x in self._list if x]
        return min(self._list)


class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, min(x, self.stack[-1][1])))

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]

    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

# ["MinStack","push","push","push","getMin","pop","top","getMin"]
# [[],[-2],[0],[-3],[],[],[],[]]

if __name__ == '__main__':
    print('min stack')

    obj = MinStack()
    for _x in [-2, 0, -3]:
        obj.push(_x)
        print(_x)
    param_1 = obj.getMin()
    print('param 1 get min:', param_1)
    obj.pop()
    param_3 = obj.top()
    print('param 3 top:', param_3)
    param_4 = obj.getMin()
    print('param 4 get min:', param_4)
