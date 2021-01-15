# reference:https://leetcode-cn.com/problems/binary-tree-inorder-traversal/solution/python3-er-cha-shu-suo-you-bian-li-mo-ban-ji-zhi-s/


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 递归
# 时间复杂度：O(n)，n为节点数，访问每个节点恰好一次。
# 空间复杂度：空间复杂度：O(h)，h为树的高度。最坏情况下需要空间O(n)，平均情况为O(logn)

# 递归1：二叉树遍历最易理解和实现版本
class Solution:
    def preorderTraversal(self, root: TreeNode) -> []:
        if not root:
            return []
        # 前序遍历
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

    def inorderTraversal(self, root: TreeNode) -> []:
        if not root:
            return []
        # 中序递归
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    def postorderTraversal(self, root: TreeNode) -> []:
        if not root:
            return []
        # 后序遍历
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]


# 递归2：通用模板，可以适应不同的题目，添加参数、增加返回条件、修改进入递归条件、自定义返回值
class SolutionGeneralRecursion:
    def preorderTraversal(self, root: TreeNode) -> []:
        def pre_order_traversal(cur):
            if not cur:
                return []
            # 前序递归
            res.append(cur.val)
            pre_order_traversal(cur.left)
            pre_order_traversal(cur.right)

        res = []
        pre_order_traversal(root)
        return res

    def inorderTraversal(self, root: TreeNode) -> []:
        def in_order_traversal(cur):
            if not cur:
                return []
            # 中序遍历
            in_order_traversal(cur.left)
            res.append(cur.val)
            in_order_traversal(cur.right)

        res = []
        in_order_traversal(root)
        return res

    def postorderTraversal(self, root: TreeNode) -> []:
        def post_order_traversal(cur):
            if not cur:
                return []
            # 后序遍历
            post_order_traversal(cur.left)
            post_order_traversal(cur.right)
            res.append(cur.val)

        res = []
        post_order_traversal(root)
        return res


# 迭代
# 时间复杂度：O(n)，n为节点数，访问每个节点恰好一次。
# 空间复杂度：O(h)，h为树的高度。取决于树的结构，最坏情况存储整棵树，即O(n)

# 迭代1：前序遍历最常用模板（后序同样可以用）
class SolutionIteration:
    def preorderTraversal(self, root: TreeNode) -> []:
        if not root:
            return []
        res = []
        stack = [root]  # stack save node info
        # 前序迭代模板：最常用的二叉树DFS迭代遍历模板
        # 根左右《-存放在栈中根右左（左先出栈）
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

    def postorderTraversal(self, root: TreeNode) -> []:
        if not root:
            return []
        res = []
        stack = [root]
        # 后序迭代，相同模板：将前序迭代进栈顺序稍作修改，最后得到的结果反转
        # 左右根《-翻转根右左《-根左右（右先出栈）
        while stack:
            cur = stack.pop()
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
            res.append(cur.val)
        return res[::-1]


# 迭代1：层序遍历最常用模板
class SolutionLevelOrder:
    def levelOrder(self, root: TreeNode) -> []:
        if not root:
            return []
        cur, res = [root], []
        while cur:
            layer, layer_val = [], []
            for _node in cur:
                layer_val.append(_node.val)
                if _node.left:
                    layer.append(_node.left)
                if _node.right:
                    layer.append(_node.right)
            cur = layer
            res.append(layer_val)
        return res


# 迭代2：前、中、后序遍历通用模板（只需一个栈的空间）
class SolutionGeneralIteration:
    def preorderTraversal(self, root: TreeNode) -> []:
        res = []
        stack = []
        cur = root
        # pre order
        # 根左右《-cur取右节点《-stack出栈赋值cur《-res入栈，stack入栈cur，cur取左节点
        while stack or cur:
            while cur:
                res.append(cur.val)
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            cur = cur.right
        return res

    def inorderTraversal(self, root: TreeNode) -> []:
        res = []
        stack = []
        cur = root
        # 中序，模板：先用指针找到每颗子树的最左下角，然后进行进出栈操作
        # 左根右《-cur取右节点《-res入栈《-stack出栈赋值cur《-stack入栈，cur取左节点
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res

    def postorderTraversal(self, root: TreeNode) -> []:
        res = []
        stack = []
        cur = root
        # post order
        # 左右根《-翻转根右左《-cur取左节点《-stack出栈赋值cur《-res入栈，stack入栈，cur取右节点
        while stack or cur:
            while cur:
                res.append(cur.val)
                stack.append(cur)
                cur = cur.right
            cur = stack.pop()
            cur = cur.left
        return res[::-1]


# 迭代3：标记法迭代（需要双倍的空间来存储访问状态）：
# 前、中、后、层序通用模板，只需改变进栈顺序或即可实现前后中序遍历，
# 而层序遍历则使用队列先进先出。0表示当前未访问，1表示已访问。
class SolutionSignIteration:
    def preorderTraversal(self, root: TreeNode) -> []:
        res = []
        stack = [(0, root)]
        # 根左右《-右左根入栈（栈存储顺序翻转）
        while stack:
            flag, cur = stack.pop()
            if not cur:
                continue
            if flag == 0:
                # 前序，标记法
                stack.append((0, cur.right))
                stack.append((0, cur.left))
                stack.append((1, cur))
            else:
                res.append(cur.val)
        return res

    def inorderTraversal(self, root: TreeNode) -> []:
        res = []
        stack = [(0, root)]
        # 左根右《-右根左入栈
        while stack:
            flag, cur = stack.pop()
            if not cur:
                continue
            if flag == 0:
                # 中序，标记法
                stack.append((0, cur.right))
                stack.append((1, cur))
                stack.append((0, cur.left))
            else:
                res.append(cur.val)
        return res

    def postorderTraversal(self, root: TreeNode) -> []:
        res = []
        stack = [(0, root)]
        # 左右根《-根右左入栈
        while stack:
            flag, cur = stack.pop()
            if not cur:
                continue
            if flag == 0:
                # 后序，标记法
                stack.append((1, cur))
                stack.append((0, cur.right))
                stack.append((0, cur.left))
            else:
                res.append(cur.val)
        return res

    def level_order(self, root: TreeNode) -> []:
        # 层序，标记法
        res = []
        queue = [(0, root)]
        while queue:
            flag, cur = queue.pop(0)  # 注意是队列，先进先出
            if not cur:
                continue
            if flag == 0:
                # 层序遍历这三个的顺序无所谓，因为是队列，只弹出队首元素
                queue.append((1, cur))
                queue.append((0, cur.left))
                queue.append((0, cur.right))
            else:
                res.append(cur.val)
        return res


# 莫里斯遍历
# 时间复杂度：O(n)，n为节点数，看似超过O(n)，有的节点可能要访问两次，实际分析还是O(n)，具体参考大佬博客的分析。
# 空间复杂度：O(1)，如果在遍历过程中就输出节点值，则只需常数空间就能得到中序遍历结果，空间只需两个指针。
# 如果将结果储存最后输出，则空间复杂度还是O(n)。

# PS：莫里斯遍历实际上是在原有二叉树的结构基础上，构造了线索二叉树，
# 线索二叉树定义为：原本为空的右子节点指向了中序遍历顺序之后的那个节点，把所有原本为空的左子节点都指向了中序遍历之前的那个节点
# emmmm，好像大学教材学过，还考过

# 此处只给出中序遍历，前序遍历只需修改输出顺序即可
# 而后序遍历，由于遍历是从根开始的，而线索二叉树是将为空的左右子节点连接到相应的顺序上，使其能够按照相应准则输出
# 但是后序遍历的根节点却已经没有额外的空间来标记自己下一个应该访问的节点，
# 所以这里需要建立一个临时节点dump，令其左孩子是root。并且还需要一个子过程，就是倒序输出某两个节点之间路径上的各个节点。
# 具体参考大佬博客

# 莫里斯遍历，借助线索二叉树中序遍历（附前序遍历）
class Solution7:
    def inorderTraversal(self, root: TreeNode) -> []:
        res = []
        # cur = pre = TreeNode(None)
        cur = root

        while cur:
            if not cur.left:
                res.append(cur.val)
                # print(cur.val)
                cur = cur.right
            else:
                pre = cur.left
                while pre.right and pre.right != cur:
                    pre = pre.right
                if not pre.right:
                    # print(cur.val) 这里是前序遍历的代码，前序与中序的唯一差别，只是输出顺序不同
                    pre.right = cur
                    cur = cur.left
                else:
                    pre.right = None
                    res.append(cur.val)
                    # print(cur.val)
                    cur = cur.right
        return res


# N叉树遍历
# 时间复杂度：时间复杂度：O(M)，其中 M 是 N 叉树中的节点个数。每个节点只会入栈和出栈各一次。
# 空间复杂度：O(M)。在最坏的情况下，这棵 N 叉树只有 2 层，所有第 2 层的节点都是根节点的孩子。
# 将根节点推出栈后，需要将这些节点都放入栈，共有 M−1个节点，因此栈的大小为 O(M)。


# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


# N叉树简洁递归
class SolutionNTRecursion:
    def preorder(self, root: 'Node') -> []:
        if not root:
            return []
        res = [root.val]
        for node in root.children:
            res.extend(self.preorder(node))
        return res


# N叉树通用递归模板
class SolutionNTRecursionTemplate:
    def preorder(self, root: 'Node') -> []:
        res = []

        def helper(root):
            if not root:
                return
            res.append(root.val)
            for child in root.children:
                helper(child)

        helper(root)
        return res


# N叉树迭代方法
class SolutionNTIteration:
    def preorder(self, root: 'Node') -> []:
        if not root:
            return []
        s = [root]
        # s.append(root)
        res = []
        while s:
            node = s.pop()
            res.append(node.val)
            # for child in node.children[::-1]:
            #     s.append(child)
            s.extend(node.children[::-1])
        return res


if __name__ == '__main__':
    # print('binary tree preorder in-order and post-order')
    # solution = Solution()
    # node = TreeNode(1)
    # node.right = TreeNode(2)
    # node.right.left = TreeNode(3)
    # preorder_result = solution.preorderTraversal(node)
    # print('preorder result:', preorder_result)
    # in_order_result = solution.inorderTraversal(node)
    # print('in-order result:', in_order_result)
    # post_order_result = solution.postorderTraversal(node)
    # print('post-order result:', post_order_result)

    # print('binary tree general recursion template')
    # solution = SolutionGeneralRecursion()
    # node = TreeNode(1)
    # node.right = TreeNode(2)
    # node.right.left = TreeNode(3)
    # pre_order_result = solution.preorderTraversal(node)
    # print('preorder result:', pre_order_result)
    # in_order_result = solution.inorderTraversal(node)
    # print('inorder result:', in_order_result)
    # post_order_result = solution.postorderTraversal(node)
    # print('postorder result:', post_order_result)

    # print('binary tree general iteration template')
    # solution = SolutionIteration()
    # node = TreeNode(1)
    # node.right = TreeNode(2)
    # node.right.left = TreeNode(3)
    # pre_order_result = solution.preorderTraversal(node)
    # print('preorder result:', pre_order_result)
    # post_order_result = solution.postorderTraversal(node)
    # print('postorder result:', post_order_result)

    # print('binary tree level order')
    # solution = SolutionLevelOrder()
    # node = TreeNode(1)
    # node.right = TreeNode(2)
    # node.right.left = TreeNode(3)
    # level_order_result = solution.levelOrder(node)
    # print('level order result:', level_order_result)

    # print('binary tree general iteration template')
    # solution = SolutionGeneralIteration()
    # node = TreeNode(1)
    # node.right = TreeNode(2)
    # node.right.left = TreeNode(3)
    # pre_order_result = solution.preorderTraversal(node)
    # print('preorder result:', pre_order_result)
    # in_order_result = solution.inorderTraversal(node)
    # print('inorder result:', in_order_result)
    # post_order_result = solution.postorderTraversal(node)
    # print('postorder result:', post_order_result)

    print('binary tree general iteration with tuple stack')
    solution = SolutionSignIteration()
    node = TreeNode(1)
    node.right = TreeNode(2)
    node.right.left = TreeNode(3)
    pre_order_result = solution.preorderTraversal(node)
    print('preorder result:', pre_order_result)
    in_order_result = solution.inorderTraversal(node)
    print('inorder result:', in_order_result)
    post_order_result = solution.postorderTraversal(node)
    print('postorder result:', post_order_result)
    level_order_result = solution.level_order(node)
    print('level order result:', level_order_result)
