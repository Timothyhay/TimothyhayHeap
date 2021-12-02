---
layout: blogpage
title: All about Linked List
comments: true
tags: LeetCode Note
---

Basic Operations:
## Merge - 21. Merge Two Sorted Lists ##

Check out the definition of linked list in C++ & Python first!

```cpp
/**
 * Definition for singly-linked list.
 */
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (!l1)
            return l2;
        else if (!l2)
            return l1;
        else if (l1->val < l2->val){
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
```

Implementation with Python:

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, l1, l2):
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```
            
And a tip:

> Practically-speaking, there is not much difference since custom comparison operators are rare. But you should use is None as a general rule.
> 
> Also, is None is a bit (~50%) faster than == None :)

This recursion approach will take O(m+n) time & space complexity, linear in the combined size of the lists.

We can also use single iteration to implement this. Use pointer `prev` to points to the current node for which we are considering adjusting its next pointer.
We increment `prev` to keep it one step behind one of our list heads.

<div class="hovereffect">
    <div class="illustration" >
        <a class="chocolat-image"  href="/images/illustration/2020-12-08/linkedlist.png"><img src="/images/illustration/2020-12-08/linkedlist.png" class="img-responsive" alt="LinkedList"></a>
        <h6>Iteration Approach. Credit: LeetCode</h6>
    </div>
</div>

p.s. There's a bug in this figure. When `l1.val == l2.val`, in this question we actually choose l2 as `prev.next`. 

More specifically, `if (l1.val < l2.val) {prev->next = l1; l1 = l1.next;}`.

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    // Here, head node rather than head pointer.
    // If both head & prev are nullptr, then they
    ListNode head = ListNode(-1);
    ListNode* prev = &head;
    while (l1 && l2){
        if (l1->val <= l2->val){
            prev->next = l1;
            l1 = l1->next;
        } else {
            prev->next = l2;
            l2 = l2->next;
        }
        prev = prev->next;
    }
    prev->next = l1 ? l1 : l2;
    return head.next;
}
```

### An interesting issue: ###

What if we define `head` as head pointer rather than a node?

Then we will get this:

    Line 18: Char 23: runtime error: member access within null pointer of type 'ListNode' (solution.cpp)
    SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior prog_joined.cpp:27:23

Because when `head` is a nullptr, `prev` will also be a nullptr. So `prev->next` is inaccessible. 
Because nullptr have no members. 

When we let `ListNode head = ListNode(-1);` or  `ListNode head = NULL;`. It has members `val(-1 or 0)` and `next(NULL)`. In such case we can manipulate `prev->next`.

    
```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(-1)
        prev = head
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        if l1 is None:
            prev.next = l2
        else:
            prev.next = l1
        return head.next
```

Time complexity: O(n+m). The while loop runs for a number of iterations equal to the sum of the lengths of the two lists.

Space complexity: O(1). The iterative approach only allocates a few pointers, so it has a constant overall memory footprint. 

## Merge - 23. Merge k Sorted Lists ##

About how to write a customized comparing function to use in the third param of `priority_queue`.
This priority queue actually stores only heads of each linked list, whose `next` may have another ListNode.

When a `head` was pop out, since each linked list was sorted, we let `head = head->next` to maintain this place with 
the next valid and reasonable value.

```cpp
struct cmp {
    bool operator()(ListNode* a, ListNode* b){
        // Small -> Large
        return a->val > b->val;
    }
};

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*, vector<ListNode*>, cmp> q;
        for (auto it : lists)
            // Use if (it) to avoid push empty vector [[]], or head->next will be nullptr while q is not empty (has [])
            if (it)
                q.push(it);

        if (q.empty())
            return nullptr;
        ListNode result;
        ListNode* head = &result;
        while (!q.empty()){
            head->next = q.top();
            head = head->next;
            q.pop();
            if (head->next)
                q.push(head->next);
        }
        return result.next;
    }
};
```


## Reverse - 206. Reverse Linked List ##

Change the current node's next pointer to point to its previous element.
 
Since a node does not have reference to its previous node, we must store its previous element `prev` beforehand. 

Also need another pointer to store the next node `nxt` before changing the reference. 

Finally return the new head reference at the end(the final `prev`, rather than final `curr`. The later one is NULL).

Implementation:

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;
        while (curr){
            ListNode* next = curr->next; // Step 1.
            curr->next = prev;           // Step 2.
            prev = curr;                 // Step 3.
            curr = next;                 // Step 4.
        }
        return prev;
    }
};
```

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr is not None:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        return prev
```

## Reorder List

https://practice.geeksforgeeks.org/problems/reorder-list/1#

Given a singly linked list: A0→A1→…→An-1→An, reorder it to: A0→An→A1→An-1→A2→An-2→…

```python
# Node Class
class Node:
    def __init__(self, val):
        self.data = val
        self.next = None


def reverseList(head):
    crt = head
    pre = None
    nxt = None
    while crt is not None:
        nxt = crt.next
        crt.next = pre
        pre = crt
        crt = nxt
    return pre


def reorderList(self):
    if self.head is None or self.head.next is None:
        return

    # 1) Find the middle point using tortoise and hare
    fast = self.head.next
    slow = self.head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

    # 2) Split the linked list in two halves
    # node1, head of first half    1 -> 2 -> 3
    # node2, head of second half   4 -> 5
    right = slow.next
    left = self.head
    # Remember to add the final None for right linklist!
    slow.next = None

    # 3) Reverse the second half, i.e., 5 -> 4
    right = reverseList(right)

    # 4) Merge alternate nodes
    self = Node(0) #Assign a dummy node and make it in-place
    crt = self

    while left is not None or right is not None:
        # First add the element from first list
        if left is not None:
            crt.next = left
            left = left.next
            crt = crt.next
        # Then add the element from second list
        if right is not None:
            crt.next = right
            right = right.next
            crt = crt.next
            
    # Assign the head of the new list to head pointer
    self = self.next

```

---

## Reference ##
[1] What is the difference between "is None" and "==None" - https://stackoverflow.com/questions/3257919/what-is-the-difference-between-is-none-and-none