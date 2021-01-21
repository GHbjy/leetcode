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
        # 中左右《-存放在栈中，顺序中右左（左先出栈）
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
        # 左右中《-翻转中右左《-中左右（右先出栈）
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
        # 中左右《-cur取右节点《-stack出栈赋值cur《-res入栈，stack入栈cur，cur取左节点
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
        # 左中右《-cur取右节点《-res入栈《-stack出栈赋值cur《-stack入栈，cur取左节点
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
        # 左右中《-翻转中右左《-cur取左节点《-stack出栈赋值cur《-res入栈，stack入栈，cur取右节点
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
        # 中左右《-右左中入栈（栈存储顺序翻转）
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
        # 左中右《-右中左入栈
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
        # 左右中《-中右左入栈
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

# 相较于中序遍历，前序遍历只需修改访问右节点时结果的输出顺序即可
# 而后序遍历，由于遍历是从根开始的，而线索二叉树是将为空的左右子节点连接到相应的顺序上，使其能够按照相应准则输出
# 但是后序遍历的根节点却已经没有额外的空间来标记自己下一个应该访问的节点，
# 所以这里需要建立一个临时节点dump，令其左孩子是root。并且还需要一个子过程，就是倒序输出某两个节点之间路径上的各个节点。
# 具体参考大佬博客

# 莫里斯遍历，借助线索二叉树中序遍历（附前序遍历）
# reference：https://blog.csdn.net/qq_22235017/article/details/108633705
class SolutionMorristraversal:
    def preorderTraversal(self, root: TreeNode) -> []:
        """
        莫里斯算法前序遍历
        莫里斯算法对于每一个节点将该节点的中序前驱节点(左子树上最右的叶子节点)的空右子树的内存指向自己
        作为从底层叶子节点返回高层根节点的途径。
        在迭代法中需要维护一个堆栈，作为从中到左，之后以此退栈实现从左子树回到根节点然后在进行向右子树的递归
        """
        res = []

        cur = root

        while cur:
            if not cur.left:  # 前序遍历是中左右的顺序，当没有左子树时，直接输出值，并转到右子树上
                res.append(cur.val)
                cur = cur.right
            else:  # 当存在左子树时，找到根节点的中序前驱节点，也就是左子树的最右的叶子
                pre = cur.left
                while pre.right and pre.right != cur:  # 判断左子树的最右叶子的右子树(本来是空内存)是否为空
                    pre = pre.right
                if not pre.right:
                    # 当右子树是空，说明该根节点是第一次被访问
                    # 按照前序，中左右，根节点第一次被访问时就应该输出
                    #
                    # 前序和中序遍历不一样的地方在于，中序遍历是左中右，是从中到左到中的时候才将根节点输出
                    # 所以中序遍历是在第二次访问节点是输出
                    #
                    # 第一次访问根节点时，将中序前驱节点的右子树空内存指向根节点
                    # 然后按照中左右的顺序，访问了根节点就需要左子树，将当前节点转向左子树，
                    res.append(cur.val)
                    pre.right = cur
                    cur = cur.left
                else:
                    # 此时中序前驱节点的右子树的本来是空的内存已经指向了根节点
                    # 说明当前的根节点，已经是从左子树访问完了又回到了根节点，第二次访问根节点了
                    # 此时就需要将中序前序节点的右子树的内存值为空，回到最初树的形状
                    # 然后当前节点转向右子树
                    pre.right = None
                    cur = cur.right
        return res

    def inorderTraversal(self, root: TreeNode) -> []:
        """
       莫里斯算法中序遍历
       莫里斯算法是将每个节点的中序的前驱节点的空右子树的空内存
       (左子树的最右叶子节点的空右子树的空右子树的空内存）指向该节点，
       作为从中序前驱节点返回上一层的途径。
       和之前的迭代法相比较，迭代法需要一个堆栈依次记录从上层到下层的记录，
       当到达最低层的节点时通过出栈的方式获得上层节点。
       而莫里斯算法，利用之前设置的空内存从底层节点返回上层节点。
        """
        res = []
        # cur = pre = TreeNode(None)
        cur = root

        while cur:
            # 当前节点无左子树则输出该节点并转到右子树
            # 因为中序遍历方式左中右，无左子树则输出中并转到右子树
            if not cur.left:
                res.append(cur.val)
                # print(cur.val)
                cur = cur.right
            else:
                # 存在左子树，对于当前节点，找中序前驱节点
                # 中序前驱节点为当前节点左子树的最右叶子节点
                pre = cur.left

                # 找当前节点的中序前驱，并且右子树不为空
                while pre.right and pre.right != cur:
                    pre = pre.right

                # 根节点的中序前驱节点是个叶子节点，叶子节点的右子树应该是None
                # 利用这个None的空内存指向该根节点
                # 通过该前驱节点的右子树是不是空，判断是不是第一次访问该根节点
                # 如果该前驱节点的右子树是空，那么是第一次访问根节点
                # 如果该前驱节点的右子树已经指向了根节点，那么该根节点已经是第二次访问
                if not pre.right:
                    # 如果是第一次访问根节点，第一次找到中序前驱
                    # 那么就将中序前驱(叶子)右子树的空内存指向该根节点
                    # 并且当前节点指向左子树
                    # 指向左子树的原因是已经记录了从底层节点返回上层的方式
                    # print(cur.val) 这里是前序遍历的代码，前序与中序的唯一差别，只是输出顺序不同
                    pre.right = cur
                    cur = cur.left
                else:
                    # 如果现在叶子节点的右子树的内存已经指向根节点了
                    # 那么就将该内存设置为空，回到树之前的样子
                    # 并且打印根节点的值
                    # 因为此时根节点已经是第二次访问了，该根节点下面的节点已经都被打印了
                    # 按照左中右的顺序，根节点的左子树下面的节点都被打印了，那么就到了对应的根节点【中】上，打印【中】，
                    # 然后转到根节点的右子树上
                    pre.right = None
                    res.append(cur.val)
                    # print(cur.val)
                    cur = cur.right
        return res

    def postorderTraversal(self, root: TreeNode) -> []:
        """
        莫里斯算法后序遍历
        莫里斯算法的后序遍历需要一个额外的空间，将该空间的左子树指向根节点
        然后依次找到每个根节点的中序前驱节点
        当第二次访问该节点时，倒叙输出从根节点左子树到前序节点的结果到最终结果上
        """
        res = []
        dump = TreeNode(None)  # 新建空节点，空节点的左子树指向根节点
        dump.left = root
        cur = dump
        while cur:
            # 当没有左子树时转向右子树
            # 因为当没有左子树时，按照后序左右中的顺序，已经没有左了，只需要考虑右和中就可以
            if not cur.left:
                cur = cur.right
            else:
                # 当存在左子树时，找到根节点的中序前驱节点
                # 并通过中序前驱节点本来应该是空的右子树判断是第几次访问根节点
                this = cur.left  # 记录左子树的节点，待会得到输出结果
                pre = cur.left
                while pre.right and pre.right != cur:
                    pre = pre.right
                if not pre.right:
                    # 当第一次访问根节点，
                    # 则将中序前驱节点的右子树指向根节点
                    # 当前节点指向左子树
                    pre.right = cur
                    cur = cur.left
                else:
                    # 如果是第二次访问根节点了
                    # 则【倒着】输出从左子树到中序节点的结果
                    # 然后转到右子树
                    #
                    # 假设是一颗2层的树
                    #          		0(临时节点)
                    #      		1
                    #  		2       3
                    # 当第二次访问根节点1 时
                    # 根结点1的中序前驱节点是2
                    # 则【倒着】输出从左子树2，到中序前驱2的路径结果，[2]
                    # 然后从根节点1到右子树3，3没有左子树然后转到右子树0(3的右节点指向0)
                    # 这是第二次访问0节点，然后【倒着】输出从左子树1到中序前序3的路径结果，[3,1]
                    # 按照这个顺序，实际上是依次获得左，然后倒着获得[右，中]（放入栈中，因此先放中再放右），
                    #
                    # 当第二次访问根节点时，已经获得了根节点左子树的全部节点，所以转到右子树
                    # 因为添加了临时节点，所以获得了根节点左子树、根节点右子树之后会回到临时节点的最高节点，
                    # 最高节点获得它的左子树的全部节点时会获得根节点的值
                    pre.right = None
                    temp = []
                    while this:
                        temp.append(this.val)
                        this = this.right
                    print(temp)
                    res.extend(temp[::-1])
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

    # print('binary tree general iteration with tuple stack')
    # solution = SolutionSignIteration()
    # node = TreeNode(1)
    # node.right = TreeNode(2)
    # node.right.left = TreeNode(3)
    # pre_order_result = solution.preorderTraversal(node)
    # print('preorder result:', pre_order_result)
    # in_order_result = solution.inorderTraversal(node)
    # print('inorder result:', in_order_result)
    # post_order_result = solution.postorderTraversal(node)
    # print('postorder result:', post_order_result)
    # level_order_result = solution.level_order(node)
    # print('level order result:', level_order_result)

    print('binary tree morris traversal')
    solution = SolutionMorristraversal()
    node = TreeNode(1)
    node.right = TreeNode(2)
    node.right.left = TreeNode(3)
    pre_order_result = solution.preorderTraversal(node)
    print('preorder result:', pre_order_result)
    in_order_result = solution.inorderTraversal(node)
    print('inorder result:', in_order_result)
    post_order_result = solution.postorderTraversal(node)
    print('postorder result:', post_order_result)
