class Solution(object):
# 1 two sums
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        num_dict = {}
        for i in range(len(nums)):
            if nums[i] in num_dict:
                return [num_dict[nums[i]], i]        
            else:
                num_dict[target - nums[i]] = i
# 2 add two numbers
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l_sum = ListNode((l1.val + l2.val)%10)
        addon = (l1.val + l2.val)//10
        l1 = l1.next
        l2 = l2.next
        l_x = l_sum
        while(l1!=None or l2!=None or addon > 0):
            if l1 != None:
                addon += l1.val
                l1 = l1.next
            if l2 != None:
                addon += l2.val
                l2 = l2.next
            thisval = addon%10
            addon = addon//10
            l_x.next = ListNode(thisval)
            l_x = l_x.next
        return l_sum
# 3 Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
            return 0
        substring_len = [1]
        l = 0
        for i in range(0, len(s)):
            l = max(i+1, l)
            for j in range(l, len(s) + 1):
                if len(set(s[i:j])) == j - i:
                    substring_len.append(j - i)
                else:
                    l = j
                    break
        return max(substring_len)
    def lengthOfLongestSubstring(self, s):
        unique_chars = set([])
        j = 0 
        n = len(s)
        longest = 0
        for i in range(n):
            while j < n and s[j] not in unique_chars:
                unique_chars.add(s[j])
                j += 1
            longest = max(longest, j - i)
            unique_chars.remove(s[i])
        return longest

# 4 Median of Two Sorted Arrays
# 5 Longest Palindromic Substring
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(set(s)) <= 1:
            return s
        isPalindrome = [[x == y for x in range(0, len(s))] for y in range(0, len(s))]
        for i in range(1, len(s)):
            for j in range(0, len(s) - i):
                if s[j] == s[j+i]:
                    if i == 1:
                        isPalindrome[j][j+i] = True
                    else:
                        if isPalindrome[j+1][j+i-1] == True:
                            isPalindrome[j][j+i] = True
        len_substring = {}
        for i in range(0, len(s)):
            for j in range(i, len(s)):
                if isPalindrome[i][j] == True:
                    if j - i in len_substring:
                        len_substring[j-i].append(s[i:j+1])
                    else:
                        len_substring[j-i] = [s[i:j+1]]
        return len_substring[max(len_substring)][0]     

# 7 Reverse Integer
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        ispos = 1
        if x < 0:
            ispos = -1
            x = -1 * x
        if x == 0:
            return 0
        strx = str(x)
        for i, x in enumerate(strx[::-1]):
            if x != 0:
                break
            else:
                pass
        ret = ispos*int(strx[::-1][i:])
        if ret > 2147483647 or ret < -2147483648:
            ret = 0
        return ret    

# 8 String to Integer (atoi)  
    def myAtoi(self, Str):
        """
        :type str: str
        :rtype: int
        """
        ret = 0
        istart = -1
        iend = 0
        if len(Str) == 0:
            return ret
        for i in range(len(Str)):
            if Str[i] == " ":
                pass
            elif Str[i] not in "0123456789-+":
                return ret
            else:
                istart = i
                break
        if istart == -1:
            return ret
        iend = len(Str)
        for i in range(istart + 1, len(Str)):
            if Str[i] not in "0123456789":
                iend = i
                break
        ret = int(Str[istart:iend] if Str[istart:iend] not in "-+" else "0")
        ret = min(2147483647, max(-2147483648, ret))
        return ret
# 9 Palindrome Number
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        to_str = str(x)
        if to_str == to_str[::-1]:
            return True
        else:
            return False
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        elif x == 0:
            return True
        elif x%10 == 0:
            return False
        digit = []
        while(x>0):
            digit.append(x%10)
            x = x//10
        if digit == digit[::-1]:
            return True
        else:
            return False
# 12 Integer to Romain
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        digits = self.num_digit(num)    
        roman = ['']*len(digits)
        dict_roman = {0:["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"], 1:["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"], 2: ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"], 3: ["", "M", "MM", "MMM"]}
        for i, a in enumerate(digits):
            roman[i] = dict_roman[i][a]
        ret = ""
        for i in roman[::-1]:
            ret += i
        return ret
            
    def num_digit(self, num):
        if num == 0:
            return [num]
        digits = []
        while num > 0:
            digits.append(num%10)
            num = num // 10
        return digits

# 14 longest common prefix
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) <= 1: 
            return "" if len(strs) == 0 else strs[0]
        minlen = min(map(len, strs))
        if minlen == 0:
            return ""
        end = 0
        ret = ""
        for i in range(minlen):
            first_letter = set(map(lambda x: x[i], strs))
            if len(set(map(lambda x: x[i], strs))) == 1:
                ret += list(first_letter)[0]
            else:
                break
        return ret

# 26 Remove Duplicates from Sorted Array
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 1:
            return len(nums)
        start = 0
        for end in range(1, len(nums)):
            if nums[start] == nums[end]:
                pass
            else:
                nums[start + 1], nums[end] = nums[end], nums[start + 1]
                start += 1
        return start + 1

# 33 Search in Rotated Sorted Array
# time 60% space 6%
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0 or (len(nums) == 1 and nums[0] != target):
            return -1
        low = 0
        high = len(nums) - 1
        mid = (low + high)//2
        while(nums[mid] != target):
            if low == high:
                return -1
            if nums[mid] > nums[low]:
                if target > nums[mid] or target < nums[low]:
                    low = mid + 1
                else:
                    high = mid
            else:
                if target < nums[mid + 1] or target > nums[high]:
                    high = mid
                else:
                    low = mid + 1
            mid = (low + high)//2
        return mid

# 53. Maximum Subarray
# 99.85%
    def maxSubArray(self, nums: List[int]) -> int:
        
        max_to_now = 0
        curr = 0
        for i in range(len(nums)):
            if curr + nums[i] > max_to_now:
                max_to_now = curr + nums[i]
                curr = max_to_now
            else:
                curr = curr + nums[i]
                if curr < 0:
                    curr = 0
        if max_to_now == 0:
            return max(nums)
        return max_to_now

# 62. Unique Paths:
# 76%
    def uniquePaths(self, m: int, n: int) -> int:
        if m <= 1 or n <= 1:
            return 1
        layer = [1]*m
        for j in range(n - 1):
            for i in range(1, len(layer)):
                layer[i] = layer[i - 1] + layer[i]
        return layer[-1]

# 63. Unique Paths II:
# 64%
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if len(obstacleGrid) == 0:
            return 0
        else:
            m = len(obstacleGrid)
            if len(obstacleGrid[0]) == 0:
                return 0
            else:
                n = len(obstacleGrid[0])
        if m <= 1 or n <= 1:
            if m == 1:
                for i in range(n):
                    if obstacleGrid[0][i] == 1:
                        return 0
            if n == 1:
                for i in range(m):
                    if obstacleGrid[i][0] == 1:
                        return 0
            return 1
        
        
        no_ob = -1
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                obstacleGrid[i][0] = no_ob
            else:
                obstacleGrid[i][0] = 0
                no_ob = 0
                if i == 0:
                    return 0
        
        no_ob = -1
        for i in range(1, n):
            if obstacleGrid[0][i] == 0:
                obstacleGrid[0][i] = no_ob
            else:
                obstacleGrid[0][i] = 0
                no_ob = 0
                
        print(obstacleGrid)
        
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    obstacleGrid[i][j] = 0
                else:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j - 1]
        
        return -obstacleGrid[m-1][n-1]
# 64. Minimum Path Sum
# 87%
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if len(grid) == 0:
            return 0
        m, n = len(grid), len(grid[0])
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        for i in range(1, n):
            grid[0][i] += grid[0][i - 1]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[m - 1][n - 1]


# 69. Sqrt(x)
# 44%
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        left = 1
        right = sys.maxsize
        while True:
            mid = left + (right - left)//2
            if mid > x/mid:
                right = mid - 1
            else:
                if mid + 1 > x/(mid + 1):
                    return mid
                left = mid + 1

# 70. Climbing Stairs
# 64%
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        if n <= 2:
            if n == 1:
                return 1
            else:
                return 2
        else:
            prev = 1
            curr = 2
            n = n - 2
            while n:
                prev, curr = curr, prev + curr
                n = n - 1
        return curr
# 74. Search a 2D Matrix
# 77%
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        if len(matrix) <= 1:
            if len(matrix) == 0 or len(matrix[0]) == 0:
                return False
        m, n = len(matrix), len(matrix[0])
        
        ibegin = 0, 0
        iend = m - 1, n - 1
        print(ibegin, iend)
        if target < matrix[ibegin[0]][ibegin[1]] or target > matrix[iend[0]][iend[1]]:
            return False
        
        while ibegin[0] < iend[0] or ibegin[0] == iend[0] and ibegin[1] <= iend[1]:
            
            mid = ((iend[0] + ibegin[0])*n + (iend[1] + ibegin[1]))//2
            imid = (mid//n, mid%n)
            print(ibegin, iend, imid, mid)
            print(matrix[imid[0]][imid[1]])
            if target == matrix[imid[0]][imid[1]]:
                return True
            elif target > matrix[imid[0]][imid[1]]:
                if imid[1] == n - 1:
                    ibegin = imid[0] + 1, 0
                else:
                    ibegin = imid[0], imid[1] + 1
            else:
                if imid[1] == 0:
                    iend = imid[0] - 1, n - 1
                else:
                    iend = imid[0], imid[1] - 1
        return False

# 78 subset
# 79% time 
    def subsets(self, nums: List[int]) -> List[List[int]]:
        results = []
        results.append([])
        for i in range(len(nums)):
            curr_n = len(results)
            for j in range(curr_n):
                tmp = results[j].copy()
                tmp.append(nums[i])
                results.append(tmp)
        return results

# 81 Search in Rotated Sorted Array II
# time 54% space 6%
    def search(self, nums: List[int], target: int) -> bool: 
        if len(nums) <= 1:
            if len(nums) == 0:
                return False
            else:
                return True if nums[0] == target else False
        
        low = 0
        high = len(nums) - 1
        
        while low <= high:
            mid = low + (high - low)//2
            if target == nums[mid]:
                return True
            else:
                if nums[mid] > nums[low]: # low to mid sorted
                    if target >= nums[low] and target < nums[mid]:
                        high = mid - 1
                    else:
                        low =  mid + 1
                elif nums[mid] < nums[low]: # mid to high sorted
                    if target > nums[mid] and target <= nums[high]:
                        low = mid + 1
                    else:
                        high = mid - 1
                else:
                    low = low + 1
        return False

# 91. Decode Ways
# 94%
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <= 1:
            return 0 if len(s) == 0 or len(s) == 1 and s[0] == '0' else 1
        if s[0] == '0':
            return 0
        
        prev1 = 1
        prev0 = 1
        curr = 1
        
        for i in range(1, len(s)):
            curr_string = int(s[i-1:i+1])
            if curr_string > 26 or curr_string < 10:
                if s[i] == '0':
                    return 0
                else:
                    curr = prev0
            else:
                if s[i] == '0':
                    curr = prev1
                else:
                    curr = prev0 + prev1
            prev1, prev0 = prev0, curr
        return curr

# 94. Binary Tree Inorder Traversal
# 79%
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        stack = []
        res = []
        curr = root
        while stack or root:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            res.append(curr.val)
            curr = curr.right
        return res
                

# 96. Unique Binary Search Trees
# 67%
    def numTrees(self, n: int) -> int:
        if n <= 1:
            return 1
        count_res  = [0]*(n + 1)
        count_res[0] = count_res[1] = 1
        
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                count_res[i] += count_res[j - 1]*count_res[i - j]
        return count_res[n]

# 98. Validate Binary Search Tree
# 55%
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        # in order traversal
        stack = []
        res = []
        while stack or root:
            
            while root:
                stack.append(root)
                root = root.left
                
            root = stack.pop()
            res.append(root.val)
            
            root = root.right
            
        print(res)
        for i in range(1, len(res)):
            if res[i] <= res[i - 1]:
                return False
        return True

# 101. Symmetric Tree
# time 68%
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isMirror(root, root)
        
    def isMirror(self, t1, t2):
        if t1 == None and t2 == None:
            return True
        if t1 == None or t2 == None:
            return False
        return t1.val == t2.val and self.isMirror(t1.left, t2.right) and self.isMirror(t1.right, t2.left)

# BFS
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None or (root.left == None and root.right == None):
            return True
        layer = []
        if root.left and root.right:
            layer = [root.left, root.right]
        else:
            return False
        
        while layer:
            l = layer.pop(0)
            r = layer.pop(0)
            
            if l.val != r.val:
                return False
            
            if l.left and r.right:
                layer.append(l.left)
                layer.append(r.right)
            elif l.left or r.right:
                return False
            
            if l.right and r.left:
                layer.append(l.right)
                layer.append(r.left)
            elif l.right or r.left:
                return False
        return True
# DFS
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        sl = [root]
        sr = [root]
        lp = root.left
        rp = root.right
        
        while(lp or sl or rp or lp):
            if not lp and rp or rp and not lp:
                return False
            if lp and rp:
                if lp.val != rp.val:
                    return False
                sl.append(lp)
                sr.append(rp)
                lp = lp.left
                rp = rp.right
            else:
                lp = sl.pop().right
                rp = sr.pop().left
        return True

# 102.Binary Tree Level Order Traversal
# 70%
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []
        queue = [root]
        res = []
        while queue:
            x = len(queue)
            for i in range(x):
                if queue[i].left:
                    queue.append(queue[i].left)
                if queue[i].right:
                    queue.append(queue[i].right)
            layer = []
            for i in range(x):
                layer.append(queue.pop(0).val)
            res.append(layer)
        return res

# 103. Binary Tree Zigzag Level Order Traversal
# time 62%
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []
        layer = [root]
        res = []
        left = 1
        while len(layer):
            
            res_layer = []
            if left:
                for i in layer:
                    res_layer.append(i.val)
                res.append(res_layer)
                left = 0
            else:
                for i in layer[::-1]:
                    res_layer.append(i.val)
                res.append(res_layer)
                left = 1
            thislayer = []
            for i in layer:
                if i.left:
                    thislayer.append(i.left)
                if i.right:
                    thislayer.append(i.right)
            layer = thislayer
        return res

# 105. Construct Binary Tree from Preorder and Inorder Traversal
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0:
            return None
        
        inorder_dict = dict(zip(inorder, range(len(inorder))))
        curr = 0
        def helperFunc(inorder_left, inorder_right):
            if inorder_left > inorder_right:
                return None
            root_val = preorder[curr]
            root = TreeNode(root_val)
            inorder_curr = inorder_dict[root_val]
            preorder.pop(0)
            root.left = helperFunc(inorder_left, inorder_curr - 1)
            root.right = helperFunc(inorder_curr + 1, inorder_right)
            return root
        return helperFunc(0, len(inorder) - 1)

               

# 112 Path Sum        
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root == None:
            return False
        if root.left == None and root.right == None and root.val == sum:
            return True
        else:
            return Solution.hasPathSum(self, root.left, sum - root.val) or Solution.hasPathSum(self, root.right, sum - root.val)

# 113. Path Sum II
# time 57%
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        
        if root == None: 
            return []
        
        res = []
        def dfs(node, val, node_list):
            if node.left == None and node.right == None and node.val + val == sum:
                node_list.append(node.val)
                res.append(node_list)
                node_list = []
            if node.left or node.right:
                node_list.append(node.val)
                if node.left:
                    dfs(node.left, val + node.val, node_list.copy())
                if node.right:
                    dfs(node.right, val + node.val, node_list.copy())
        dfs(root, 0, [])
        return res
  
# 118. Pascal's Triangle
# 72% 
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        retList = [[0]*i for i in range(1, numRows + 1)]
        for i in range(numRows):
            retList[i][0] = retList[i][-1] = 1
            for j in range(1, i):
                retList[i][j] = retList[i - 1][j - 1] + retList[i - 1][j]
        return retList

#120. Triangle
# 98%
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        if n <= 1:
            return 0 if n == 0 else triangle[0][0]
    
        path = triangle[-1].copy()
        for i in range(n - 2, -1, -1):
            for j in range(0, i + 1):
                path[j] = triangle[i][j] + min(path[j], path[j + 1])
        return path[0]


# 121. Best Time to Buy and Sell Stock
# 99%
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        
        max_profit = 0
        buy_time = 0
        for i in range(1, len(prices)):
            if prices[i] - prices[buy_time] > max_profit:
                max_profit = prices[i] - prices[buy_time]
            if prices[i] < prices[buy_time]:
                buy_time = i
        return max_profit 

# 123. Best Time to Buy and Sell Stock III
# 85%
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0   
        max_left = [0]*len(prices) # max_left[i]: max profit in prices[:i+1] if can only buy once
        max_right = [0]*len(prices) # max_right[i]: max profit in prices[i+1:] if can only buy once
        buy_time = 0
        max_profit_left = 0
        
        sell_time = len(prices) - 1
        max_profit_right = 0
        
        for i in range(1, len(prices)):
            if prices[i] - prices[buy_time] > max_profit_left:
                max_profit_left = prices[i] - prices[buy_time]
            if prices[i] < prices[buy_time]:
                buy_time = i
            max_left[i] = max_profit_left
            
            if prices[sell_time] - prices[len(prices) - i - 1] > max_profit_right:
                max_profit_right = prices[sell_time] - prices[len(prices) - i - 1]
            if prices[sell_time] < prices[len(prices) - i - 1]:
                sell_time = len(prices) - i - 1
            max_right[len(prices) - i - 1] = max_profit_right

        max_profit = 0
        for i in range(len(prices)):
            if max_right[i] + max_left[i] > max_profit:
                max_profit = max_right[i] + max_left[i]
        return max_profit

# 99%
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        
        # if len(prices) == 1:
        #     return 0        
        
# two transactions in total. the first one is followed by the first one. for each transaction, find the min price and update the max profit.         
                
        minPrice1, minPrice2 = prices[0], prices[0]
        maxProfit1, maxProfit2 = 0 , 0
        
        for i in prices:
            if i < minPrice1:
                minPrice1 = i
                
            if i - minPrice1 > maxProfit1:
                maxProfit1 = i - minPrice1
# the actual price of second buy is (i - maxProfit1)             
            if i - maxProfit1 < minPrice2:
                minPrice2 = i - maxProfit1
                
            if i - minPrice2 > maxProfit2:
                maxProfit2 =  i - minPrice2
                
        return maxProfit2
                
# 144. Binary Tree Preorder Traversal
# 
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        res = []
        curr = root
        stack = []
        while stack or curr:
            while curr:
                stack.append(curr)
                res.append(curr.val)
                curr = curr.left
            curr = stack.pop().right
        return res

# 145. Binary Tree Postorder Traversal
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root == None:
            return []
        res = []
        stack = []
        curr = root 
        while stack or curr:
            print(res)
            
            while curr:
                if curr.right:
                    stack.append(curr.right)
                stack.append(curr)
                curr = curr.left
                
            curr = stack.pop()
            
            if stack and curr.right and stack[-1] == curr.right:
                stack.pop()
                stack.append(curr)
                curr = curr.right
            else:
                res.append(curr.val)
                curr = None
        return res
        


# 153. Find Minimum in Rotated Sorted Array
# time 49% space 5%
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        low = 0
        high = len(nums) - 1
        mid = low + (high - low)//2
        
        while 1:
            if low == mid:
                return nums[high] if nums[high] < nums[low] else nums[low]
                
            if nums[low] > nums[high]:
                if nums[low] > nums[mid]:
                    if nums[mid] > nums[mid - 1]:
                        high = mid - 1
                    else:
                        return nums[mid]
                else:
                    #nums[low] < nums[mid]
                    if nums[mid] < nums[mid+1]:
                        low = mid + 1
                    else:
                        return nums[mid + 1]
                mid = low + (high - low)//2
            else:
                return nums[low]

# 198. House Robber
# 72%
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 0:
            return 0 if len(nums) == 0 else nums[0]
        max_gain = [0]*(len(nums) + 1)
        last = 0
        for i in range(1, len(nums) + 1):
            if last == 0:
                max_gain[i] = max_gain[i - 1] + nums[i - 1]
                last = 1
            else:
                max_gain[i] = max(max_gain[i - 1], max_gain[i - 2] + nums[i - 1])
        return max_gain[-1]

# 199 Binary Tree Right Side View
# time 56%
    def rightSideView(self, root: TreeNode) -> List[int]:
        if root == None:
            return []
        rightview = []
        thislayer = [root]
        while(len(thislayer)):
            rightview.append(thislayer[-1].val)
            layer_len = len(thislayer)
            for i in range(layer_len):
                layer_size = len(thislayer)
                if thislayer[i].left:
                    thislayer.append(thislayer[i].left)
                if thislayer[i].right:
                    thislayer.append(thislayer[i].right)
                
            for i in range(layer_len):
                thislayer.pop(0)
        return rightview

# 207 Course Schedule
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        pre = [[] for i in range(numCourses)]
        visit = [0]*numCourses
        for i in prerequisites:
            pre[i[0]].append(i[1])
        
        def dfs(i):
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            visit[i] = -1
            for j in pre[i]:
                if not dfs(j):
                    return False
            visit[i] = 1
            return True
        
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True

# 238. Product of Array Except Self
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [1]*len(nums)
        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]
        right = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            res[i] = right*res[i]
            right *= nums[i]
        return res

# 268 Missing Number
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return int(len(nums)*(len(nums) + 1)/ 2 - sum(nums))

# 278  First Bad Version
# time 38% space 6%
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        if isBadVersion(1):
            return 1
        low = 1
        high = n
        mid = low + (high - low)//2
        while(mid != low):
            if isBadVersion(mid):
                high = mid
            else:
                low = mid
            mid = low + (high - low)//2
        return high

# 283 Move Zeroes
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1:
            return
        if len(nums) == 2:
            if nums[0] == 0:
                nums[0], nums[1] = nums[1], nums[0]
                return
        i1 = 0
        i2 = 1
        while(i2 < len(nums)):
            if nums[i1] == 0 and nums[i2] != 0:
                nums[i1], nums[i2] = nums[i2], nums[i1]
            if nums[i1] != 0:
                i1 += 1
            i2 += 1
        return


# 326. Power of Three
# 98%
    def isPowerOfThree(self, n: int) -> bool:
        if n < 1:
            return False
        while n%3 == 0:
            n = n/3
        return n == 1

# 344. Reverse String
# 58%
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        for i in range(len(s)//2):
            s[i], s[-1-i] = s[-1- i], s[i]

# 378 Kth Smallest Element in a Sorted Matrix
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        low, high = matrix[0][0], matrix[n - 1][n - 1]
        while low < high:
            mid = low + (high - low)//2
            temp = sum([self.binary_search(y, mid, n) for y in matrix])
            if temp < k:
                low = mid + 1
            else:
                high = mid
        return low
    
    def binary_search(self, row, x, n):
        l, r = 0, n
        while l < r:
            mid = l + (r - l) // 2
            if row[mid] <= x:
                l = mid + 1
            else:
                r = mid
        return l  

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        low, high = matrix[0][0], matrix[n - 1][n - 1]
        while low < high:
            mid = low + (high - low)//2
            count = self.count_lower_than_mid(matrix, n, mid)
            if count < k:
                low = mid + 1
            else:
                high = mid
        return low
    
    def count_lower_than_mid(self, matrix, n, x):
        count = 0
        i = n - 1
        j = 0
        while i >= 0 and j < n:
            if matrix[i][j] <= x:
                j += 1
                count += i + 1
            else:
                i -= 1
        return count

# 387. First Unique Character in a String
# 72%
    def firstUniqChar(self, s: str) -> int:
        if len(s) <= 0:
            return -1 if len(s) == 0 else 0
        dict_check = {}
        for i in range(len(s)):
            if s[i] in dict_check:
                dict_check[s[i]][0] = False
            else:
                dict_check[s[i]] = [True, i]
        min_test = -1 
        for i in dict_check:
            if dict_check[i][0]:
                if min_test == -1:
                    min_test = dict_check[i][1]
                else:
                    if min_test > dict_check[i][1]:
                        min_test = dict_check[i][1]
        return min_test

# 410. Split Array Largest Sum
# time 60%
    def splitArray(self, nums: List[int], m: int) -> int:
        # range of the return value
        min_sum = max(nums)
        max_sum = max(sum(nums), min_sum)
        
        while min_sum < max_sum:
            mid = min_sum + (max_sum - min_sum)//2
            if self.checkValid(nums, mid, m):
                max_sum = mid
            else:
                min_sum = mid + 1
        return min_sum
        
    def checkValid(self, nums, mid, m):
        count = 0
        curr = 0
        for i in nums:
            curr += i
            if curr > mid:
                count += 1
                if count >= m:
                    return False
                curr = i
        return True

# 412. Fizz Buzz
# 58%
    def fizzBuzz(self, n: int) -> List[str]:
        res = [str(i + 1) for i in range(n)]
        for i in range(1, n//3+1):
            res[i*3-1] = 'Fizz'
        for i in range(1, n//5+1):
            res[5*i-1] = 'Buzz'
        for i in range(1, n//15+1):
            res[15*i-1] = 'FizzBuzz'
        return res

# 513. Find Bottom Left Tree Value
# time 80%
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        layer = [root]
        while(len(layer)):
            thislayer = []
            for i in layer:
                if i.left:
                    thislayer.append(i.left)
                if i.right:
                    thislayer.append(i.right)
            if thislayer:
                layer = thislayer
            else:
                return layer[0].val

# 559. Maximum Depth of N-ary Tree
# time 38%
    def maxDepth(self, root: 'Node') -> int:
        if root == None:
            return 0
        layer = [root]
        max_depth = 0
        while(len(layer)):
            max_depth += 1
            layer_size = len(layer)
            for i in range(layer_size):
                for j in range(len(layer[i].children)):
                    layer.append(layer[i].children[j])
            for i in range(layer_size):
                layer.pop(0)
        return max_depth

# 690. Employee Importance
# time 74%
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        id_dict = {i.id: i for i in employees}
        layer = [id_dict[id]]
        total_importance = 0
        while layer:
            thislayer = []
            for i in layer:
                total_importance += i.importance
                for j in i.subordinates:
                    thislayer.append(id_dict[j])
            layer = thislayer
        return total_importance

# 744. Find Smallest Letter Greater Than Target
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        low = 0
        high = len(letters) - 1
        mid = low + (high - low)//2
        while low < high:
            if letters[mid] > target:
                high = mid
            elif letters[mid] <= target:
                low = mid + 1
            mid = low + (high - low)//2           
        return letters[high] if letters[high] > target else letters[0]

# 876. Middle of the Linked List
    def middleNode(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        node1 = head
        node2 = head
        while node2.next.next != None:
            node1, node2 = node1.next, node2.next.next
            if node2.next == None:
                break
        if node2.next == None:
            return node1
        return node1.next


# 988. Smallest String Starting From Leaf
    def smallestFromLeaf(self, root: TreeNode) -> str:

        def val_to_letter(n):
            return chr(n + ord('a'))

        all_str = []

        def dfs(node, strlist):
            if node == None:
                return
            if node.left == node.right == None:
                all_str.append((strlist + val_to_letter(node.val))[::-1])
                return
            else:
                strlist = strlist + val_to_letter(node.val)
                dfs(node.left, strlist)
                dfs(node.right, strlist)
        dfs(root, '')

        return min(all_str)

