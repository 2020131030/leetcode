# leetcode add two numbers
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        answer_lst = ListNode()
        curr_answer = answer_lst
        
        curr_l1 = l1
        curr_l2 = l2
  
        while True:
            if not curr_l1 and not curr_l2:
                break
            
            _sum = curr_answer.val

            if curr_l1:
                _sum += curr_l1.val
                curr_l1 = curr_l1.next
            
            if curr_l2:
                _sum += curr_l2.val
                curr_l2 = curr_l2.next
            
            remainder = _sum % 10
            share = _sum // 10
            
            curr_answer.val = remainder
            
            if share or curr_l1 or curr_l2:
                curr_answer.next = ListNode(share)  
                curr_answer = curr_answer.next

        return answer_lst

---------------------------------------


# two sum

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        length = len(nums)
    
        
        for idx1, num1 in enumerate(nums):
            for idx2 in range(idx1+1, length):
                num2 = nums[idx2]
                _sum = num1 + num2 
                
                if _sum == target:
                    return [idx1, idx2] 

---------

# Logest Palindromic Substring

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def lookup(s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left+1:right]
        if len(s) <= 1 or s == s[::-1]: return s
        ans = ''
        for i in range(len(s)-1):
            ans = max(ans, lookup(s, i, i+1), lookup(s, i, i+2), key=len)
        return ans


----------

# Zigzag Conversion

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1: return s
        group = (numRows-1)*2
        ans = ''
        for j in range(0, numRows):
            for i, v in enumerate(s):
                if i % group == j or i % group == group - j:
                    ans += v
        return ans


----------

# Palindrome Number

class Solution:
    def isPalindrome(self, x: int) -> bool:
        ret=None
        
        if x < 0:
            ret=False
        else:
            original=x
            reverse=0
            while x:
                remainder=x%10
                x=x//10
                reverse=reverse*10+remainder

            if reverse == original:            
                ret=True
            else:
                ret=False
        
        return ret


--------

# Container With Most Water

class Solution:
    def maxArea(self, height: List[int]) -> int:
        
        left=0
        right=len(height)-1
        max_area=0
        while (right-left>0) :
            max_area=max(max_area,(right-left)*min(height[left],height[right]))
                
            if height[left]>=height[right]: # The right is shorter than left
                right-=1
            else: # The left is shorter than right
                left+=1
            
        return max_area


-------

# Roman to Integer

class Solution:
    def romanToInt(self, s: str) -> int:
        lst = {"I": 1, "V" : 5, "X":10, "L":50, "C":100, "D":500, "M":1000}
        
        i = 0
        word = 0

        # one letter
        if len(s) <= 1:
            word = lst[s]

        # two letters
        elif len(s) == 2:
            if lst[s[i]] >= lst[s[i+1]]:
                word = lst[s[i]] + lst[s[i+1]]
            else:
                word = lst[s[i+1]] - lst[s[i]]

        # more than 3 letters
        else:
            while i < len(s) - 1 :
                if lst[s[i]] >= lst[s[i+1]]:
                    word += lst[s[i]]
                    i += 1
                else:
                    word += lst[s[i+1]] - lst[s[i]]
                    i += 2

            # sum last word
            if lst[s[-2]] >= lst[s[-1]]:
                word += lst[s[-1]]
            else:
                pass
        
        return word



------

# Longest Common Prefix

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        answer = ''
        strs.sort()
        for i in range(len(strs[0])):
            curr = strs[0][i]
            if self.includesCharacter(strs,curr,i):
                answer += curr
            else:
                break
        return answer
            
    def includesCharacter(self, strs, curr, i):
        for x in strs[1:]:
            if x[i] == curr:
                continue
            else:
            	return False
        return True


-------

# 3Sum

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        results = []
        nums.sort()
        
    
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
        
            left, right = i + 1, len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum < 0:
                    left += 1
                elif sum > 0:
                    right -= 1
                else:
                    results.append([nums[i], nums[left], nums[right]])
                
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
        return results


------

# Letter Combination of a Phone Number

class Solution(object):
    def letterCombinations(self, digits):
        numDict = {"2":["a","b","c"], "3":["d","e","f"], "4":["g","h","i"], "5":["j","k","l"], "6":["m","n","o"], "7":["p","q","r","s"], "8":["t","u","v"], "9":["w","x","y","z"]}
        resultList = []

        for i in range(len(digits)):
            outputList = [] 
            if resultList == []: 
               resultList = numDict[digits[0]]
               continue
            for k in resultList: 
                    for p in numDict[digits[i]]:
                        outputList.append(k+p) 
            resultList = outputList
        return resultList

-----

# Valid Parentheses

class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        brackets={'}':'{',')':'(',']':'['}
        for bracket in s:
            if bracket in brackets.values(): 
                stack.append(bracket)
            else:
                if stack and brackets[bracket]==stack[-1] :  
                    stack.pop()
                else: 
                    return False
        
        if stack:
            return False
        return True

------

# Merge Two Sorted Lists

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        
        head=ListNode(-1)
        cursor=head
        
        while l1!=None and l2!=None:
            if l1.val <= l2.val:
                cursor.next=l1 
                l1=l1.next 
            else:
                cursor.next=l2 
                l2=l2.next 
            
            cursor=cursor.next 
        
        
    
        if l1!=None:
            cursor.next=l1
        
        else:
            cursor.next=l2
            
        return head.next


-------


# Generate Paretheses

class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        ans = []
        
        def dfs(left, right, s):
            if len(s) == n * 2:
                ans.append(s)
                return
            if left < n:
                dfs(left + 1, right, s + "(")
            if left > right:
                dfs(left, right + 1, s + ")")
        
        dfs(0, 0, "")
        return ans


-------

# Remove Duplicates from Sorted Array

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums)==0: return 0
        
        
        i=0 
        for j in range(1,len(nums)):
            if nums[i]!=nums[j]: 
                i+=1  
                nums[i]=nums[j] 
        
        return i+1


------

# Remove Element

class Solution:
    def removeElement(self, nums, val):
        for i in range(len(nums)):
            if nums[0] != val:
                nums.append(nums[0])
                del nums[0]
            else:
                del nums[0]
        return len(nums)


------

# Find the Index of the First Occurence in a String

class Solution:
  def strStr(self, haystack: str, needle: str) -> int:
    m = len(haystack)
    n = len(needle)

    for i in range(m - n + 1):
      if haystack[i:i + n] == needle:
        return i

    return -1



-----


# Divide Two Integers

class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        ans = dividend/divisor
        
        if ans < -(2**31): ans = -(2**31)
        elif ans > 2**31-1 : ans = 2**31-1
            
        if ans < 0:     ans = ceil(ans)
        elif ans > 0:   ans = floor(ans)
            
        return int(ans)



-----


# Next Permutation

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        flag = True
        for i in range(len(nums)-1, 0, -1):
            if nums[i] > nums[i-1]:
                index = i-1
                flag = False
                break
        if flag: nums.sort()
        else:
            swapNum = float('inf')
            swapIndex = -1
            for i in range(index+1, len(nums)):
                if nums[i] > nums[index] and swapNum > nums[index]:
                    swapNum = nums[i]
                    swapIndex = i
            temp = nums[swapIndex]
            nums[swapIndex] = nums[index]
            nums[index] = temp 
            tail = sorted(nums[index+1:])
            nums[:] = nums[:index+1] + tail



---------


# Search in Rotated Sorted Array

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + ((right - left) // 2)
            # find target
            if target == nums[mid]:
                return mid
            
            # left portion
            if nums[left] <= nums[mid]:
                if target > nums[mid] or target < nums[left]:
                    left = mid + 1
                else:
                    right = mid - 1
            
            # right portion
            else:
                if target < nums[mid] or target > nums[right]:
                    right = mid - 1
                else:
                    left = mid + 1
            
        return -1


-------

# Find First and Last Position of Element in Sorted Array


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if target not in nums:
            return [-1, -1]
        l = len(nums) - 1

        left, right = 0, l
        start, end = -1, -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                start = mid
                right = mid - 1
            else:
                left = mid + 1

        end = start
        left = start + 1
        right = l
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                end = mid
                left = mid + 1
            else:
                right = mid - 1

        return [start, end]



------



# Search Insert Position

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        
        left=0
        right=len(nums)-1
        while left<=right:
            mid=(left+right)//2
            if nums[mid]>target: 
                right=mid-1
            elif nums[mid]<target: 
                left=mid+1   
            else: 
                return mid
        
        return left 


-----

# Valid Sudoku

def checkcorrect(board: List[List[str]], row, col):
    c = set()
    r = set()
    b = set()
    for i in board[row]:
        if i == '.': continue
        if i in r: return False
        r.add(i)
    for j in range(9):
        if board[j][col] == '.': continue
        if board[j][col] in c: return False
        c.add(board[j][col])
    trow = (row // 3) * 3
    tcol = (col // 3) * 3
    for i in range(trow, trow + 3):
        for j in range(tcol, tcol + 3):
            if board[i][j] == '.': continue
            if board[i][j] in b: return False
            b.add(board[i][j])
    return True

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.' and checkcorrect(board, i, j) == False: return False
        return True



------


# Count and Say

class Solution:
    def countAndSay(self, n: int) -> str:
        res = ""
        ct = 1 
        
        if n == 1:
            return "1"
        
        prev =  self.countAndSay(n - 1) 
        
        for i in range(len(prev)):
            if  i == len(prev) - 1 or prev[i] != prev[i + 1]:
                res += str(ct) + prev[i]
                ct = 1
            else :
                ct +=1
                
        return res


-----

# Combination Sum

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(curr, csum):
            if csum > target:  
                return
            if csum == target:  
                result.append(path[:])
                return

            for val in curr:
                path.append(val)
                dfs(candidates[candidates.index(val) :], sum(path))
                path.pop()

        path, result = [], []
        dfs(candidates, 0)
        return result



------

# Jump Game 2

class Solution(object):
    def jump(self, nums):
        
        maxList = [nums[i] + i for i in range(len(nums))]
        startIdx = 0
        move = 0 
        cnt = 0

        while move < len(nums)-1: 
            cnt += 1
            startIdx, move = move+1, max(maxList[startIdx:move+1])

        return cnt


------

# Permutations

from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.nums = nums
        self.perms = []
        
        self.bt([])
        return self.perms
    
    def bt(self, crnt_set: List[int]):
        if len(crnt_set) == len(self.nums):
            self.perms.append(crnt_set.copy())
            return
        
        for num in self.nums:
            if num in crnt_set:
                continue
            
            crnt_set.append(num)
            self.bt(crnt_set)
            crnt_set.pop()




------


# Rotate Image

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        N = len(matrix)
        if N == 1: return

        
        l, r = 0, N-1
        while l < r:
            matrix[l], matrix[r] = matrix[r], matrix[l]
            l += 1
            r -= 1

        
        for i in range(N):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]



-------


# Group Anagrams


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = collections.defaultdict(list)
        
        for word in strs:
            anagrams[''.join(sorted(word))].append(word)
        return list(anagrams.values())



------


# Maximum Subrray

class Solution:
   def maxSubArray(self, nums: List[int]) -> int:
       best_sum = -sys.maxsize
       cur_sum = 0
       
       for num in nums:
           cur_sum = max(num, num + cur_sum)
           best_sum = max(cur_sum, best_sum)
       
       return best_sum

------


# Spiral Matrix

class Solution:
  def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    if not matrix:
      return []

    m = len(matrix)
    n = len(matrix[0])
    ans = []
    r1 = 0
    c1 = 0
    r2 = m - 1
    c2 = n - 1


    while len(ans) < m * n:
      j = c1
      while j <= c2 and len(ans) < m * n:
        ans.append(matrix[r1][j])
        j += 1
      i = r1 + 1
      while i <= r2 - 1 and len(ans) < m * n:
        ans.append(matrix[i][c2])
        i += 1
      j = c2
      while j >= c1 and len(ans) < m * n:
        ans.append(matrix[r2][j])
        j -= 1
      i = r2 - 1
      while i >= r1 + 1 and len(ans) < m * n:
        ans.append(matrix[i][c1])
        i -= 1
      r1 += 1
      c1 += 1
      r2 -= 1
      c2 -= 1

    return ans


-----


# Jump Game

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) -1

        for i in range(len(nums)-1, -1, -1):
            if i + nums[i] >= goal:
                goal = i
            
        return True if goal == 0 else False



-------


# Merge Intervals


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        n = len(intervals)
        if n == 1:
            return [intervals[0]]
        
        intervals = sorted(intervals)

        i = 0
        while i < len(intervals):
            while i < len(intervals) - 1 and intervals[i][1] >= intervals[i + 1][0]:
                intervals[i][1] = max(intervals[i][1], intervals[i + 1][1])
                intervals.pop(i + 1)
                       
            i += 1

        return intervals



-------


# Insert Interval

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        ans = []

        for i in range(len(intervals)):
            if intervals[i][0] > newInterval[1]:
                ans.append(newInterval)
                return ans + intervals[i:]
            elif intervals[i][1] < newInterval[0]:
                ans.append(intervals[i])
            else:
                newInterval = [min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])]

        ans.append(newInterval)
        return ans 




-----


# Length of Last Word


class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        sList = s.split(' ')
        sList = [string for string in sList if not string == '']
        return len(sList[-1])


-------


# Rotate List


class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next:   return head
        nhead = head
        count = 1
        
        while nhead.next:
            count += 1
            nhead = nhead.next
        k %= count
        k = count - k
        if k == count:  return head
        
        nhead = head
        while k > 1:
            k -= 1
            nhead = nhead.next
        
        ans = nhead.next
        nhead.next = None
        
        findTail = ans
        
        while findTail.next:
            findTail = findTail.next
        findTail.next = head
        
        return ans



-----

# Unique Paths


class Solution(object):
    def uniquePaths(self, m, n):
        dp = [[1] * n for i in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[m - 1][n - 1]



-----

# Minimum Path Sum


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        minCost2d = a = [[0] * cols for i in range(rows)]
        
        
        minCost2d[0][0] = grid[0][0]
        for colIdx in range(1,cols):
            minCost2d[0][colIdx] = grid[0][colIdx] + minCost2d[0][colIdx-1]
        for rowIdx in range(1,rows):
            minCost2d[rowIdx][0] = grid[rowIdx][0] + minCost2d[rowIdx-1][0]
        
        
        for rowIdx in range (1,rows):
            for colIdx in range (1,cols):
                prevCol = colIdx - 1
                prevRow = rowIdx - 1
                
                upCost = minCost2d[prevRow][colIdx]
                leftCost = minCost2d[rowIdx][prevCol]
                
                prevCost = min(upCost,leftCost)
                cost = prevCost + grid[rowIdx][colIdx]        
                minCost2d[rowIdx][colIdx] = cost
                    
        minCost = minCost2d[rows-1][cols-1]    
        return minCost



------


# Plus One

class Solution:
    def plusOne(self, digits):
        # n = (int(''.join(map(str, digits)))) + 1
        # return list(map(int, list(str(n))))
        flag = False
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
                flag = True
            else:
                digits[i] += 1
                flag = False
                break
        if flag:
            digits = [1] + digits
        return digits



-----

# Add Binary

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return format(int(a,2) + int(b,2), 'b')



-------

# Sqrt(x)

import math

class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        
        while left <= right :
            mid = (left + right) // 2 
            calc = mid * mid
            if calc == x :
                return mid
            elif calc < x :
                left = mid + 1
            else :
                right = mid - 1
        else :
            return right



-----


# Climbing Stairs


from typing import List

class Solution:
    def climbStairs(self, n: int) -> int:
        dp_array = [0,1,2]

        if n < len(dp_array):
            return dp_array[n]

        for i in range(3, n+1):
            ith_way = dp_array[i-1] + dp_array[i-2]
            dp_array.append(ith_way)

        return dp_array[n]



------


# Simplify Path

class Solution:
    def simplifyPath(self, path: str) -> str:
        path = path.split('/')
        
        while '' in path:
            path.remove('')
        while '.' in path:
            path.remove('.')
        
        while '..' in path:
            idx = path.index('..')
            del path[idx]
            if idx > 0:
                del path[idx-1]
        
        path = '/'.join(path)
        
        return '/' + path



-----


# Edit Distance

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0 for _ in range(len(word2)+1)] for _ in range (len(word1)+1)]

        dp[0][0] = 0
        
        for i in range(1, len(word1)+1):
            dp[i][0] = i
            
        for i in range(1, len(word2)+1):
            dp[0][i] = i
            
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]
                else: dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)
        
        return dp[len(word1)][len(word2)]



-----


# Set Matrix Zeroes

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row, col = set(), set()
        M, N = len(matrix), len(matrix[0])
        for i in range(M):
            for j in range(N):
                if matrix[i][j] == 0:
                    row.add(i)
                    col.add(j)
        
        for r in row:
            for i in range(N):
                matrix[r][i] = 0
        
        for c in col:
            for j in range(M):
                matrix[j][c] = 0

                





    



<img width="275" alt="스크린샷 2024-06-21 오전 12 38 26" src="https://github.com/2020131030/leetcode/assets/169224394/438dd6f6-973d-4c6f-bdb3-b41e511e1294">

