from typing import List

class ListNode:
    def __init__(self, data):
        self.val = data 
        self.next = None 

class Solution:
    def __init__(self, l1, l2):
        self.list1, len_of_list1 = self.get_linked_list_representation(l1)
        self.list2, len_of_list2 = self.get_linked_list_representation(l2)
    
    def get_linked_list_representation(self, l: list) -> (ListNode, int):
        # Time Complexity: O(N), where N is the number of nodes in the list
        if l is not None and len(l) > 0:
            current = ListNode(-1)
            head = current 
            for data in l:
                current.next = ListNode(data)
                current = current.next 
            head = head.next 
            return head, len(l)
        return None, 0

    def print_linked_list(self, node: ListNode, do_not_print: bool = True) -> List: 
        if node:
            data = []
            current = node
            while current != None:
                data.append(current.val)
                current = current.next 
            if not do_not_print:
                print(f"data in list : {data}")
            return data 
        return []

    def add_two_linked_lists(self) -> ListNode:
        if self.list1 is None:
            return self.list2
        if self.list2 is None:
            return self.list1        
        
        dummy_head = ListNode(0)
        current = dummy_head
        carry = 0
        p = self.list1
        q = self.list2
        
        # Traverse both lists and add corresponding digits along with any carry
        while p is not None or q is not None or carry != 0:
            sum_val = carry
            
            if p is not None:
                sum_val += p.val
                p = p.next
                
            if q is not None:
                sum_val += q.val
                q = q.next
                
            carry = sum_val // 10
            current.next = ListNode(sum_val % 10)
            current = current.next
            
        return dummy_head.next     


def add_two_numbers(l1, l2):
    # l1 and l2 are lists of digits in reverse order
    sol = Solution(l1, l2)
    result_head = sol.add_two_linked_lists()
    result_list = sol.print_linked_list(result_head)
    return result_list
    