class Solution:
    def isValid(self, s: str) -> bool:
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''

    def isValid_2(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False

        pairs = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        stack = list()
        for ch in s:
            if ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:
                stack.append(ch)

        return not stack


if __name__ == '__main__':
    print('is valid ')
    solution = Solution()
    # print('none:', solution.isValid(''))
    # print('empty:', solution.isValid('()'))
    # print('valid:', solution.isValid('{}[]()((([])))'))
    # print('not valid:', solution.isValid('[]{)'))

    print('none:', solution.isValid_2(''))
    print('empty:', solution.isValid_2('()'))
    print('valid:', solution.isValid_2('{}[]()((([])))'))
    print('not valid:', solution.isValid_2('[]{)'))
