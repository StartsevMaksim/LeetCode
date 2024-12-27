from collections import deque
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
def generateTree(string_val):
    if not string_val:
        return None
    node_val_arr = string_val.split(',')    
    node_val_arr = [int(val) if val != 'null' else None for val in node_val_arr]
    node_val_iter = iter(node_val_arr)

    root = TreeNode(next(node_val_iter))
    bfs_queue = deque([root])
    while True:
        try:
            cur_node = bfs_queue.popleft()
            left_val = next(node_val_iter)
            if left_val is not None:
                cur_node.left = TreeNode(left_val)
                bfs_queue.append(cur_node.left)
            right_val = next(node_val_iter)
            if right_val is not None:
                cur_node.right = TreeNode(right_val)
                bfs_queue.append(cur_node.right)
        except StopIteration:
            return root

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def generateList(value_string):
    value_arr = list(map(lambda x: int(x), value_string.split(',')))
    head = ListNode(value_arr[0])
    cur_val = head
    for val in value_arr[1:]:
        cur_val.next = ListNode(val)
        cur_val = cur_val.next
    return head

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
def generateNodeList(val_string):
    vals = val_string.split(',')
    vals = [int(val) if val != 'null' else None for val in vals]
    root = Node(vals[0])
    node = root
    for val in vals[1:]:
        node.children = Node(val)
        node = node.children
    return root