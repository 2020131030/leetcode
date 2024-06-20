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


<img width="275" alt="스크린샷 2024-06-21 오전 12 38 26" src="https://github.com/2020131030/leetcode/assets/169224394/438dd6f6-973d-4c6f-bdb3-b41e511e1294">

