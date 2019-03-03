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

# 410. Split Array Largest Sum

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
