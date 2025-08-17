#Пары чисел
#Здесь ещё и сортировка слиянием
class NicePairs(object):
    def merge(self, num_arr, start_index_1, finish_index_1, start_index_2, finish_index_2):
     index_1 = start_index_1
        index_2 = start_index_2
        temp = list(range(finish_index_2 - start_index_1 + 1))
        for temp_index in range(finish_index_2 - start_index_1 + 1):
            if (index_1 <= finish_index_1) and (index_2 <= finish_index_2):
                if num_arr[index_1] < num_arr[index_2]:
                    temp[temp_index] = num_arr[index_1]
                    index_1 += 1
                else:
                    temp[temp_index] = num_arr[index_2]
                    index_2 += 1
            elif index_1 > finish_index_1:
                temp[temp_index] = num_arr[index_2]
                index_2 += 1
            else:
                temp[temp_index] = num_arr[index_1]
                index_1 += 1
        temp_index = 0
        for num_index in range(start_index_1, finish_index_2 + 1):
            num_arr[num_index] = temp[temp_index]
            temp_index += 1

    def mergeSort(self, num_arr, start_index, finish_index):           
        if finish_index - start_index > 0:   
            middle_index = start_index + int((finish_index - start_index) / 2)
            self.mergeSort(num_arr, start_index, middle_index)
            self.mergeSort(num_arr, middle_index + 1, finish_index)
            self.merge(num_arr, start_index, middle_index, middle_index + 1, finish_index)
            
    def rev(self, x):
        st = str(x)
        return int(st[::-1])
        
    def countNicePairs(self, nums):
        res_arr = [x - self.rev(x) for x in nums]
        self.mergeSort(res_arr, 0, len(res_arr) - 1)
        total_count = 0
        prev_value = res_arr[0]
        count = 1
        print(res_arr)
        for index in range(1, len(res_arr)):
            if prev_value == res_arr[index]:
                count += 1
            if (prev_value != res_arr[index]) or (index == len(res_arr) - 1):
                total_count += count * (count - 1) / 2
                count = 1
            
            prev_value = res_arr[index]
        return int(total_count % (10 ** 9 + 7))

#Последовательность диагональных элементов
#Записываем в Map
class DiagonalOrder(object):
    def findDiagonalOrder(self, nums):
        res_arr = []
        res_map = {}
        indexSum_arr = [[x + index for x in range(len(nums[index]))] for index in range(len(nums))]
        for index in range(len(indexSum_arr)):
            for key in indexSum_arr[index]:
                if key in res_map:
                    val = res_map[key]
                    val.append(nums[index][key - index])
                    res_map[key] = val
                else:
                    res_map[key] = [nums[index][key - index]] 
        for key in range(len(res_map)):
            val = res_map[key]
            val.reverse()
            res_arr.extend(val)
        return res_arr

#С помощью поиска в ширину
class DiagonalOrder(object):
    def findDiagonalOrder(self, nums):
        res_arr = []
        bfs_queue = [(0, 0)]
        while len(bfs_queue) > 0:
            node_x, node_y = bfs_queue.pop(0)
            if (len(nums) > node_x + 1) and (len(nums[node_x + 1]) > node_y) and ((node_x + 1, node_y) not in bfs_queue):
                bfs_queue.append((node_x + 1, node_y))
            if (len(nums[node_x]) > node_y + 1) and ((node_x, node_y + 1) not in bfs_queue):
                bfs_queue.append((node_x, node_y + 1))
            res_arr.append(nums[node_x][node_y])
        return res_arr

#Арифметические последовательности
class ArithmeticSubarrays(object):
    def checkArithmeticSubarrays(self, nums, l, r):
        res_arr = []
        for index in range(len(l)):
            subArr = nums[l[index] : r[index] + 1]
            if len(subArr) == 1:
                res_arr.append(True)
            else:
                theor_sum = (min(subArr) + max(subArr)) * len(subArr) / 2
                fact_sum = sum(subArr)
                if theor_sum != fact_sum:
                    res_arr.append(False)
                else:
                    res = True
                    subArr.sort()
                    d = subArr[1] - subArr[0]
                    for sub_index in range(2, len(subArr)):
                        if subArr[sub_index] - subArr[sub_index - 1] != d:
                            res = False
                    res_arr.append(res)
        return res_arr

#Максимальная сумма монет
class Coins(object):
    def maxCoins(self, piles):
        arr = piles
        arr.sort()
        res = 0
        index = len(arr) - 2
        for i in range(int(len(arr) / 3)):
            res += arr[index]
            index -= 2
        return res

#Максимальная подматрица
class Submatrix(object):
    def largestSubmatrix(self, matrix):
        height_matrix = matrix
        for index_col in range(len(height_matrix[0])):
            height = 0
            for index_row in range(len(height_matrix)):
                if height_matrix[index_row][index_col] == 1:
                    height += 1
                else:
                    height = 0
                height_matrix[index_row][index_col] = height
        for index_row in range(len(height_matrix)):
            
        return height_matrix

#Кол-во единиц
class HammingWeight(object):
    def hammingWeight(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        return (n % 2) + self.hammingWeight(n // 2)
        

#Сумма единиц
class MinPartitions(object):
    def minPartitions(self, n):
        return int(max(n))

#Минимум перестановок
class MinimumMoves(object):   
    def minimumMoves(self, grid):
        steps_map = {}
        val_arr = []
        for zero_x in range(len(grid)):
            for zero_y in range(len(grid)):
                if grid[zero_x][zero_y] == 0:
                    zero_steps_arr = []
                    for pos_x in range(len(grid)):
                        for pos_y in range(len(grid)):
                            if grid[pos_x][pos_y] > 1:
                                zero_steps_arr.append(abs(pos_x - zero_x) + abs(pos_y - zero_y))
                    steps_map[(zero_x, zero_y)] = zero_steps_arr
                elif grid[zero_x][zero_y] > 1:
                    val_arr.append(grid[zero_x][zero_y])
        
        total_steps_arr = []
        for zero_x in range(len(grid)):
            for zero_y in range(len(grid)):
                if grid[zero_x][zero_y] == 0:
                    zero_steps_arr = steps_map[(zero_x, zero_y)]
                    if len(total_steps_arr) == 0:
                        for index in range(len(zero_steps_arr)):
                            total_steps_elem_key = zero_steps_arr[index]
                            total_steps_elem_val = list(val_arr)
                            total_steps_elem_val[index] -= 1
                            total_steps_arr.append({total_steps_elem_key: total_steps_elem_val})
                    else:
                        temp_arr = []
                        for index in range(len(zero_steps_arr)):    
                            for total_steps_elem in total_steps_arr:
                                total_steps_elem_key = list(total_steps_elem.keys())[0]
                                total_steps_elem_val = total_steps_elem[total_steps_elem_key]
                                if total_steps_elem_val[index] > 1:
                                    temp_elem_key = total_steps_elem_key + zero_steps_arr[index]
                                    temp_elem_value = list(total_steps_elem_val)
                                    temp_elem_value[index] -= 1
                                    temp_arr.append({temp_elem_key: temp_elem_value})
                        total_steps_arr = temp_arr
        steps = [list(elem.keys())[0] for elem in total_steps_arr]
        return min(steps)

#Минимальная сумма
class MinPossibleSum(object):    
    def minimumPossibleSum(self, n, target):
        modulo = 10 ** 9 + 7
        a_1 = 1
        a_N = (target // 2) if (target // 2) < n else n
        length = a_N
        sum = (a_1 + a_N) * a_N // 2
        sum = sum % modulo
        if length == n:
            return sum
        sum += target
        sum = sum % modulo
        length += 1
        rem_length = n - length
        sum += (2 * target + 1 + rem_length) * rem_length // 2
        return sum % modulo

#Проверка на равенство 2-х строк
class StringsAreEqual(object):
    def arrayStringsAreEqual(self, word1, word2):
        iter_1 = iter(word1)
        cur_elem_1 = next(iter_1)
        cur_index_1 = 0
        endFlag_1 = False
        
        iter_2 = iter(word2)
        cur_elem_2 = next(iter_2)
        cur_index_2 = 0
        endFlag_2 = False
        
        while (not endFlag_1) and (not endFlag_2):
            if cur_elem_1[cur_index_1] != cur_elem_2[cur_index_2]:
                return False
            cur_index_1 += 1
            cur_index_2 += 1
            if cur_index_1 == len(cur_elem_1):
                try:
                    cur_elem_1 = next(iter_1)
                    cur_index_1 = 0
                except:
                    endFlag_1 = True
            if cur_index_2 == len(cur_elem_2):
                try:
                    cur_elem_2 = next(iter_2)
                    cur_index_2 = 0
                except:
                    endFlag_2 = True
        return endFlag_1 and endFlag_2               
            

#Массив палиндромов
class Palindrome(object):
    def kthPalindrome(self, queries, intLength):
        def makePalindrome(position, firstPartLength, oddFlag):
            firstPart = str(10 ** (firstPartLength - 1) + (position - 1))
            return (-1 if (len(firstPart) > firstPartLength) 
                       else int(firstPart + firstPart[-(1 + int(oddFlag)): :-1]))
        oddFlag = bool(intLength % 2)
        fPLength = int(oddFlag) + intLength // 2 
        return [makePalindrome(pos, fPLength, oddFlag) for pos in queries]

#Подстрока с максимальным знаением
class GoodInteger(object):
    def largestGoodInteger(self, num):
        max_value = '0'
        for index in range(2, len(num)):
            if (num[index - 2] == num[index - 1] == num[index]) and (num[index] > max_value):
                max_value = num[index]
        return max_value * 3

#Число матчей
class NumberOfMatches(object):
    def numberOfMatches(self, n):
        return 0 if n == 1 else  n // 2 + self.numberOfMatches(n // 2 + n % 2)

#Банк
class TotalMoney(object):
    def totalMoney(self, n):
        fullWeeks = n // 7
        lastWeek = (n - fullWeeks * 7) % 7
        fullWSum = 7 * (fullWeeks ** 2 + 7 * fullWeeks) // 2
        lastWSum = lastWeek * (2 * fullWeeks + lastWeek + 1) // 2
        return fullWSum + lastWSum

class LargestOddNumber(object):
    def largestOddNumber(self, num):
        res = ''
        for index in range(1, len(num) + 1):
            if int(num[-index]) % 2 == 1:
                index = -(index - 1) if index > 1 else len(num)
                res = num[:index]
                return res
        return res

#Дерево
class Tree(object):
    def tree2str(self, root):
        level = 0
        dfs_stack = [(root, level)]
        res_arr = []
        while len(dfs_stack) > 0:
            cur_node, cur_level = dfs_stack.pop(-1)
            res_arr.append((str(cur_node.val), cur_level))
            level = cur_level + 1
            if cur_node.right is not None:
                dfs_stack.append((cur_node.right, level))
                if cur_node.left is not None:
                    dfs_stack.append((cur_node.left, level))
                else:
                    temp_node = TreeNode('')
                    dfs_stack.append((temp_node, level))
            else:
                if cur_node.left is not None:
                    dfs_stack.append((cur_node.left, level))
        res_str, prev_level = res_arr.pop(0)
        for index in range(len(res_arr)):
            val, level = res_arr[index]
            if prev_level < level:
                res_str += '('
            elif prev_level == level:
                res_str += ')('
            else:
                for i in range(prev_level - level):
                    res_str += ')'
                res_str += ')('
            res_str += val
            if index == len(res_arr) - 1:
                for i in range(level):
                    res_str += ')'
            else:
                prev_level = level
        return res_str

#Подстрока макс. длины
class LongestSubstring(object):
    def lengthOfLongestSubstring(self, s):
        repeat_dict = {}
        max_length = 0
        temp_length = 0
        for index in range(len(s)):
            temp_length += 1
            elem = ord(s[index])
            if elem not in repeat_dict:
                repeat_dict[elem] = index
            else:
                temp_length = min(temp_length, index - repeat_dict[elem])
                repeat_dict[elem] = index
            max_length = max(max_length, temp_length)
        return max_length
                

#Число, встречающиеся с вероятностью больше 0.25
class SpecialInteger(object):
    def findSpecialInteger(self, arr):
        min_length = (len(arr) // 4) + 1
        for index in range(len(arr)):
            if arr[index + min_length - 1] == arr[index]:
                return arr[index]

#Маршрут до городов
class DestCity(object):
    def destCity(self, paths):
        node_arr = set()
        for elem in [x[0] for x in paths]:
            node_arr.add(elem)
        for elem in [x[1] for x in paths]:
            if elem not in node_arr:
                return elem

#Палиндром
class LongestPalindrome(object):
    def longestPalindrome(self, s):
        def checkPol(s, index, max_length):
            if (s[index:index + max_length] != s[index:index + max_length][::-1]) or (index + max_length > len(s)):
                return max_length
            return checkPol(s, index, max_length + 1)
        pol_turple = (0, 1)
        max_length = 1
        for index in range(len(s)):
            temp_length = max(max_length, checkPol(s, index, max_length))
            if max_length < temp_length:
                max_length = temp_length
                pol_turple = (index, index + max_length - 1)
        return s[pol_turple[0]:pol_turple[1]]        

#1913. Maximum Product Difference Between Two Pairs
class ProductDifference(object):
    def maxProductDifference(self, nums):
        first_max_index = 0
        second_max_index = 1
        first_min_index = 2
        second_min_index = 3
        for index in range(len(nums)):
            if nums[index] > nums[first_max_index]:
                second_max_index = first_max_index
                first_max_index = index
            elif (nums[index] > nums[second_max_index]) and (index != first_max_index):
                second_max_index = index
            if nums[index] < nums[first_min_index]:
                second_min_index = first_min_index
                first_min_index = index
            elif (nums[index] < nums[second_min_index]) and (index != first_min_index):
                second_min_index = index
        return nums[first_max_index] * nums[second_max_index] - nums[second_min_index] * nums[first_min_index]

#661. Image Smoother
class ImageSmoother(object):
    def imageSmoother(self, img):
        smooth_matrix = [list(x) for x in img]
        for index_row in range(len(img)):
            for index_col in range(len(img[0])):
                div = 1
                for x in filter(lambda x: 0 <= x < len(img), range(index_row - 1, index_row + 2)):
                    for y in filter(lambda y: 0 <= y < len(img[0]), range(index_col - 1, index_col + 2)):
                        if not (x == index_row and y == index_col):
                            smooth_matrix[index_row][index_col] += img[x][y]
                            div += 1
                smooth_matrix[index_row][index_col] //= div 
        return smooth_matrix

#1637. Widest Vertical Area Between Two Points Containing No Points
class WidthOfVerticalArea(object):
    def maxWidthOfVerticalArea(self, points):
        x_arr = [x[0] for x in points]
        x_arr.sort()
        x_iter = iter(x_arr)
        width_arr = list(map(lambda x: x - next(x_iter), x_arr[1:]))
        return max(width_arr)

#310. Minimum Height Trees НЕ РЕШЕНО
class MinHeightTrees(object):
    def findMinHeightTrees(self, n, edges):
        def makeEdgeDict(edges):
            edges_dict = {}
            for edge in edges:
                node_0, node_1 = edge[0], edge[1]
                if node_0 not in edges_dict:
                    edges_dict[node_0] = set()
                edges_dict[node_0].add(node_1)
                if node_1 not in edges_dict:
                    edges_dict[node_1] = set()
                edges_dict[node_1].add(node_0)
            return edges_dict

        if n == 1:
            return [0]
        edges_dict = makeEdgeDict(edges)
        leaf_nodes = [node for node in edges_dict if len(edges_dict[node]) == 1]
        while len(edges_dict) > 2:
            temp_leaf_nodes = []
            for leaf_node in leaf_nodes:
                neighbour_node = edges_dict.pop(leaf_node).pop()
                edges_dict[neighbour_node].remove(leaf_node)
                if len(edges_dict[neighbour_node]) == 1:
                    temp_leaf_nodes.append(neighbour_node)
            leaf_nodes = temp_leaf_nodes
        return list(edges_dict.keys())

#1422. Maximum Score After Splitting a String
class MaxScore(object):
    def maxScore(self, s):
        max_sum = 0
        for index in range(1, len(s) - 1):
            zero_sum = index - sum([int(x) for x in s[:index] if int(x) > 0])
            ones_sum = sum([int(x) for x in s[index:] if int(x) > 0])
            max_sum = max(max_sum, zero_sum + ones_sum)
        return max_sum
            

#938. Range Sum of BST
class Solution(object):
    def rangeSumBST(self, root, low, high):
        def getNodeVal(node, sum):
            if low <= node.val <= high:
                sum[0] += node.val
            if node.left is not None:
                getNodeVal(node.left, sum)
            if node.right is not None:
                getNodeVal(node.right, sum)
        sum = [0]
        getNodeVal(root, sum)
        return sum[0]

#872. Leaf-Similar Trees
class Solution(object):
    def leafSimilar(self, root1, root2):
        def makeNodeSeq(node, res_arr):
            if node.left is not None:
                makeNodeSeq(node.left, res_arr)
            if node.right is not None:
                makeNodeSeq(node.right, res_arr)
            if (node.left is None) and (node.right is None):
                res_arr.append(node.val)
        root1_seq = []
        makeNodeSeq(root1, root1_seq)
        root2_seq = []
        makeNodeSeq(root2, root2_seq)
        return root1_seq == root2_seq

#2385. Amount of Time for Binary Tree to Be Infected
class Solution(object):
    def amountOfTime(self, root, start):
        def addEdge(cur_node, next_node, edges):
            edges[cur_node].add(next_node)
            edges[next_node] = set([cur_node])
        def makeEdges(root, edges):
            dfs_queue = deque([root])
            while dfs_queue:
                cur_node = dfs_queue.popleft()
                if cur_node.left:
                    dfs_queue.append(cur_node.left)
                    addEdge(cur_node.val, cur_node.left.val, edges)
                if cur_node.right:
                    dfs_queue.append(cur_node.right)
                    addEdge(cur_node.val, cur_node.right.val, edges)
        def findMaxHeight(start, edges):
            dfs_queue = deque([start])
            dfs_checked = set()
            height = -1
            while dfs_queue:
                level = len(dfs_queue)
                while level > 0:
                    cur_node = dfs_queue.popleft()
                    dfs_queue.extend(edges[cur_node] - dfs_checked)
                    dfs_checked.add(cur_node)
                    level -= 1
                height += 1
            return height
        edges = {root.val: set()}
        makeEdges(root, edges)
        return findMaxHeight(start, edges)
                

#1704. Determine if String Halves Are Alike
class Solution(object):
    def halvesAreAlike(self, s):
        lowels = set('aeiou')
        first_part = s[:len(s)//2].lower()
        second_part = s[len(s)//2:].lower()
        return len([x for x in first_part if x in lowels]) == len([x for x in second_part if x in lowels])

#2225. Find Players With Zero or One Losses
class FindWinners(object):
    def findWinners(self, matches):
        winner_set = set()
        loser_set = set()
        one_loss_set = set()
        for match in matches:
            winner, loser = match[0], match[1]
            winner_set.add(winner)
            if loser not in loser_set:
                loser_set.add(loser)
                one_loss_set.add(loser)
            else:
                one_loss_set.discard(loser)
        answer_0 = list(winner_set - loser_set)
        answer_1 = list(one_loss_set)
        answer_0.sort()
        answer_1.sort()
        return [answer_0, answer_1]

#1207. Unique Number of Occurrences
class UniqueOccurrences(object):
    def uniqueOccurrences(self, arr):
        num_dict = {}
        for num in arr:
            if num not in num_dict:
                num_dict[num] = 0
            else:
                num_dict[num] += 1
        occ_arr = num_dict.values()
        occ_set = set(occ_arr)
        return len(occ_arr) == len(occ_set)

#70. Climbing Stairs
class ClimbStairs(object):
    def climbStairs(self, n):
        def placement(m, k):
            return factorial(m) // (factorial(k) * factorial(m - k))
        k = 0
        result = 0
        while k * 2 <= n:
            m = n - k
            result += placement(m, k)
            k += 1
        return result

#931. Minimum Falling Path Sum
class MinFallingPathSum(object):
    def minFallingPathSum(self, matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        sum_row = matrix[0]
        for row in matrix[1:]:
            temp_row = []
            for index in range(len(sum_row)):
                if index == 0:
                    temp_row.extend([sum_row[0] + row[0], sum_row[0] + row[1]])
                else:
                    temp_row[index-1] = min(sum_row[index] + row[index-1], temp_row[index-1])
                    temp_row[index] = min(sum_row[index] + row[index], temp_row[index])
                    if index < len(sum_row) - 1:
                        temp_row.append(sum_row[index] + row[index+1])
            sum_row = temp_row
        return min(sum_row)

#645. Set Mismatch
class ErrorNums(object):
    def findErrorNums(self, nums):
        cur_set = set()
        full_set = set(range(1, len(nums) + 1))
        res_arr = []
        for num in nums:
            if num in cur_set:
                res_arr.append(num)
            cur_set.add(num)
        res_arr.append((full_set - cur_set).pop())
        return res_arr

#1239. Maximum Length of a Concatenated String with Unique Characters
class MaxLength(object):
    def maxLength(self, arr):
        def check_connect(target, string):
            concated_string = target
            if target != string:
                concated_string += string
            for index_char in range(len(concated_string) - 1):
                if concated_string[index_char] in concated_string[index_char+1:]:
                    return False
            return True
            
        connect_matrix = []
        for target in arr:
            connect_row = []
            for string in arr:
                connect_row.append(check_connect(target, string))
            connect_matrix.append(connect_row)

        length_arr = [len(string) for string in arr]
        max_length = 0
        for index_row in range(len(connect_matrix)):
            connect_row = list(connect_matrix[index_row])
            for index_make_false in range(len(connect_row)):
                if not connect_row[index_make_false]:
                    continue
                concated_index_arr = []
                for index_col in range(len(connect_matrix)):
                    if connect_row[index_col]:
                        concated_flag = True
                        for concated_index in concated_index_arr:
                            if not connect_matrix[index_col][concated_index]:
                                concated_flag = False
                        if concated_flag:
                            concated_index_arr.append(index_col)
                if concated_index_arr:
                    max_length = max(max_length, sum([length_arr[index] for index in concated_index_arr]))
                if index_row != index_make_false:
                    connect_row[index_make_false] = False
        return max_length

#1457. Pseudo-Palindromic Paths in a Binary Tree
class PseudoPalindromicPaths(object):
    def pseudoPalindromicPaths(self, root):
        dfs_stack = deque([(None, root)])
        palindromic_amount = 0
        checked_arr = []
        end_flag = False
        while dfs_stack:
            mother_root, cur_node = dfs_stack.pop()
            if end_flag:
                palindromic_amount += int(isPalindromic([node.val for node in checked_arr]))
                checked_arr = checked_arr[:checked_arr.index(mother_root)+1]
                end_flag = False    
            if cur_node.right:
                dfs_stack.append((cur_node, cur_node.right))
            if cur_node.left:    
                dfs_stack.append((cur_node, cur_node.left))
            checked_arr.append(cur_node)
            if not cur_node.right and not cur_node.left:
                end_flag = True 
        palindromic_amount += int(isPalindromic([node.val for node in checked_arr]))
        return palindromic_amount

#279. Perfect Squares
class NumSquares(object):
    def numSquares(self, n):
        sqr_dict = {}
        for cur_num in range(1, n + 1):
            cur_sqrt = sqrt(cur_num)
            if cur_sqrt == int(cur_sqrt):
                sqr_dict[cur_num] = 1
                for num in range(1, n):
                    new_num = cur_num + num
                    if new_num > n:
                        break
                    if new_num not in sqr_dict:
                        sqr_dict[new_num] = new_num
                    else:
                        sqr_dict[new_num] = min(sqr_dict[cur_num] + sqr_dict[num], sqr_dict[new_num])
        return sqr_dict[n]
        
                

#739. Daily Temperatures
class DailyTemperatures(object):
    def dailyTemperatures(self, temperatures):
        temp_stack = deque()
        res_arr = [0 for index in range(len(temperatures))]
        for index, temp in enumerate(temperatures):
            while temp_stack and temperatures[temp_stack[-1]] < temp:
                temp_index = temp_stack.pop()
                res_arr[temp_index] = index - temp_index
            temp_stack.append(index)
        return res_arr

#1291. Sequential Digits
class SequentialDigits(object):
    def sequentialDigits(self, low, high):
        def digitAmount(num):
            amount = 0
            while num:
                num //= 10
                amount += 1
            return amount
        def builtFirstNumOnDigit(digit):
            res_num = 0
            num = 1
            while digit:
                res_num += num * (10 ** (digit - 1))
                num += 1
                digit -= 1
            return res_num
        def increaseNum(num, digit):
            res_num = 0
            while digit:
                res_num += 1 * (10 ** (digit - 1))
                digit -= 1
            return num + res_num
        res_arr = []
        cur_digit = digitAmount(low)
        cur_num = builtFirstNumOnDigit(cur_digit)
        while cur_num <= high:
            if cur_num >= low:
                res_arr.append(cur_num)
            cur_num = increaseNum(cur_num, cur_digit)
            if cur_num % 10 == 0:
                cur_digit += 1
                cur_num = builtFirstNumOnDigit(cur_digit)
        return res_arr

#387. First Unique Character in a String
class FirstUniqChar(object):
    def firstUniqChar(self, s):
        char_amount = {}
        for char in s:
            if char in char_amount:
                char_amount[char] += 1
            else:
                char_amount[char] = 1
        for index, char in enumerate(s):
            if char_amount[char] == 1:
                return index
        return -1

#49. Group Anagrams
class GroupAnagrams(object):
    def groupAnagrams(self, strs):
        group_dict = {}
        for index, string in enumerate(strs):
            string_list = list(string)
            string_list.sort()
            sorted_string = ''.join(string_list)
            if sorted_string in group_dict:
                group_dict[sorted_string].append(index)
            else:
                group_dict[sorted_string] = [index]
        res_arr = [[strs[index] for index in group_dict[key]] for key in group_dict.keys()]
        return res_arr

#451. Sort Characters By Frequency
class FrequencySort(object):
    def frequencySort(self, s):
        letter_dict = {}
        for char in s:
            if char in letter_dict:
                letter_dict[char] += 1
            else:
                letter_dict[char] = 1
        amount_dict = {}
        for key in letter_dict:
            amount = letter_dict[key]
            if amount in amount_dict:
                amount_dict[amount].append(key * amount)
            else:
                amount_dict[amount] = [key * amount]
        sorted_keys = sorted(amount_dict.keys(), reverse=True)
        res_str = ''
        for key in sorted_keys:
            res_str += ''.join(amount_dict[key])
        return res_str

#76. Minimum Window Substring
class MinWindow(object):
    def minWindow(self, s, t):
        target_letters_amount = {}
        for char in t:
            if char in target_letters_amount:
                target_letters_amount[char] += 1
            else:
                target_letters_amount[char] = 1
        search_flag = False
        target_letters_index = deque([])
        letters_amount = sum(target_letters_amount.values())
        max_left, max_right = 0, len(s) - 1
        for cur_right, char in enumerate(s):
            if char in target_letters_amount:
                target_letters_amount[char] -= 1
                target_letters_index.append(cur_right)
                if target_letters_amount[char] >= 0:
                    letters_amount -= 1
            if letters_amount == 0:
                search_flag = True
                while letters_amount == 0:
                    cur_left = target_letters_index.popleft()
                    if (cur_right - cur_left) < (max_right - max_left):
                        max_left, max_right = cur_left, cur_right
                    target_letters_amount[s[cur_left]] += 1
                    if target_letters_amount[s[cur_left]] > 0:
                        letters_amount += 1
        return s[max_left: max_right+1] if search_flag else '' 

#368. Largest Divisible Subset
class LargestDivisibleSubset(object):
    def largestDivisibleSubset(self, nums):       
        nums.sort()
        connect_dict = {nums[0]: (None, 0)}
        max_num, max_length = nums[0], 0
        for index, num in enumerate(nums[1:], 1):
            connect_prev_num, connect_length = None, 0
            for prev_num in nums[index-1::-1]:
                if (num % prev_num == 0) and (connect_length <= connect_dict[prev_num][1]):
                    connect_prev_num, connect_length = prev_num, connect_dict[prev_num][1] + 1
                    if connect_length > max_length:
                        max_num = num
                        max_length = connect_length
            connect_dict[num] = (connect_prev_num, connect_length)
        res_arr = [max_num]
        next_num = connect_dict[max_num][0]
        while next_num:
            res_arr.append(next_num)
            next_num = connect_dict[next_num][0]
        return res_arr
            

#169. Majority Element
class MajorityElement(object):
    def majorityElement(self, nums):
        nums.sort()
        return nums[len(nums) // 2]

#2108. Find First Palindromic String in the Array
class FirstPalindrome(object):
    def firstPalindrome(self, words):
        for word in words:
            if word == ''.join(reversed(word)):
                return word
        return ''

#2149. Rearrange Array Elements by Sign
class RearrangeArray(object):
    def rearrangeArray(self, nums):
        pos_arr = []
        neg_arr = []
        for num in nums:
            if num > 0:
                pos_arr.append(num)
            else:
                neg_arr.append(num)
        res_arr = []
        for index, pos_num in enumerate(pos_arr):
            res_arr.append(pos_num)
            res_arr.append(neg_arr[index])
        return res_arr

#22. Generate Parentheses
class Parenthesis(object):
    def generateParenthesis(self, n):
        parenthesis_dict = {(0, -2): '))',
                            (2, 0): '((',
                            (1, -1): '()',
                            (-1, 1): ')('}
        strings = [('', 0)]
        for index in range(n):
            temp_strings = []
            for string, degree in strings:
                for l_degree, r_degree in parenthesis_dict:
                    new_degree = degree + l_degree
                    if new_degree >= 0:
                        new_degree += r_degree
                        if 0 <= new_degree <= 2 * (n - index - 1):
                            temp_strings.append((string + parenthesis_dict[(l_degree, r_degree)], new_degree))
            strings = temp_strings
        return [string[0] for string in strings]

#1481. Least Number of Unique Integers after K Removals
class LeastNumOfUniqueInts(object):
    def findLeastNumOfUniqueInts(self, arr, k):
        num_dict = {}
        for num in arr:
            if num not in num_dict:
                num_dict[num] = 1
            else:
                num_dict[num] += 1
        freq_dict = {}
        for num in num_dict:
            freq = num_dict[num]
            if freq not in freq_dict:
                freq_dict[freq] = freq
            else:
                freq_dict[freq] += freq
        amount = 1 
        while k:
            if amount not in freq_dict:
                amount += 1
            else:
                freq_dict[amount] -= 1
                if not freq_dict[amount]:
                    freq_dict.pop(amount)
                k -= 1
        res = 0
        for amount in freq_dict:
            temp_res = freq_dict[amount] // amount
            if amount * temp_res != freq_dict[amount]:
                temp_res += 1
            res += temp_res
        return res

#2971. Find Polygon With the Largest Perimeter
class LargestPerimeter(object):
    def largestPerimeter(self, nums):
        nums.sort()
        side_amount = 0
        max_perim = -1
        temp_perim = 0
        for num in nums:
            side_amount += 1
            if (side_amount >= 3) and (temp_perim > num):
                temp_perim += num
                max_perim = max(temp_perim, max_perim)
            else:
                temp_perim += num
        return max_perim

#231. Power of Two
class PowerOfTwo(object):
    def isPowerOfTwo(self, n):
        k = math.log2(n)
        return k % 1 == 0.0

#268. Missing Number
class MissingNumber(object):
    def missingNumber(self, nums):
        n = len(nums)
        theor_sum = n * (n + 1) // 2
        imper_sum = sum(nums)
        if imper_sum == theor_sum:
            return 0
        else:
            return theor_sum - imper_sum

#1642. Furthest Building You Can Reach
class FurthestBuilding(object):
    def furthestBuilding(self, heights, bricks, ladders):          
        heights_dif = [heights[index+1] - num for index, num in enumerate(heights[:-1])]
        ladders_arr = []
        min_ladders_arr = max(heights_dif)
        res_index = 0
        for index, num in enumerate(heights_dif):
            if index == len(heights_dif) - 1:
                res_index = index
            if num > 0:
                if len(ladders_arr) < ladders:
                    min_ladders_arr = min(min_ladders_arr, num)
                    ladders_arr.append(num)
                    continue
                if num <= min_ladders_arr:
                    bricks -= num
                else:
                    bricks -= min_ladders_arr
                    ladders_arr.remove(min_ladders_arr)
                    ladders_arr.append(num)
                    min_ladders_arr = min(ladders_arr)
                if bricks < 0:
                    res_index = index - 1
                    break
        return res_index + 1

#997. Find the Town Judge
class FindJudge(object):
    def findJudge(self, n, trust):
        if n == 1:
            return 1
        degree_dict = {}
        for edge in trust:
            start_node, end_node = edge[0], edge[1]
            if start_node not in degree_dict:
                degree_dict[start_node] = 0
            if end_node not in degree_dict:
                degree_dict[end_node] = 0
            degree_dict[start_node] -= 1
            degree_dict[end_node] += 1
        for key in degree_dict:
            if degree_dict[key] == n - 1:
                return key
        return -1

#4. Median of Two Sorted Arrays
class MedianSortedArrays(object):
    def findMedianSortedArrays(self, nums1, nums2):
        len_1, len_2 = len(nums1), len(nums2)
        access_1, access_2 = bool(len_1), bool(len_2)
        index_1, index_2 = 0, 0
        res_arr = []
        while index_1 + index_2 <= (len_1 + len_2) // 2:
            if access_1 and access_2:
                if nums1[index_1] < nums2[index_2]:
                    res_arr.append(nums1[index_1])
                    index_1 += 1
                    if index_1 == len_1:
                        access_1 = False
                else:
                    res_arr.append(nums2[index_2])
                    index_2 += 1
                    if index_2 == len_2:
                        access_2 = False
            elif access_1:
                res_arr.append(nums1[index_1])
                index_1 += 1
            else:
                res_arr.append(nums2[index_2])
                index_2 += 1
            res_arr = res_arr[-2:]
        return res_arr[-1] if (len_1 + len_2) % 2 == 1 else (res_arr[0] + res_arr[1]) / 2

#1. Two Sum
class TwoSum(object):
    def twoSum(self, nums, target):
        num_index_dict = {}
        for index, num in enumerate(nums):
            if num not in num_index_dict:
                num_index_dict[num] = []
            num_index_dict[num].append(index)
        print(num_index_dict)
        for first_index, first_num in enumerate(nums):
            second_num = target - first_num
            if second_num in num_index_dict:
                for second_index in num_index_dict[second_num]:
                    if first_index != second_index:
                        return [first_index, second_index]

#787. Cheapest Flights Within K Stops 
class CheapestPrice(object):
    def findCheapestPrice(self, n, flights, src, dst, k):
        def makeCityToCityDict(flights, dst):
            city_dict = {}
            for flight in flights:
                start_city, end_city, price = flight[0], flight[1], flight[2]
                if start_city != dst:
                    if start_city not in city_dict:
                        city_dict[start_city] = [True, [[end_city], [price]]]
                    else:
                        city_dict[start_city][1][0].append(end_city)
                        city_dict[start_city][1][1].append(price)
            return city_dict

        city_dict = makeCityToCityDict(flights, dst)
        price_dict = {}
        bfs_queue = deque([(src, 0, 0)])
        while bfs_queue:
            flight = bfs_queue.popleft()
            cur_city, cur_price, cur_stop = flight[0], flight[1], flight[2]
            if cur_stop <= k + 1:
                if cur_city not in price_dict:
                    price_dict[cur_city] = cur_price
                else:
                    if price_dict[cur_city] > cur_price:
                        price_dict[cur_city] = cur_price
                        if cur_city in city_dict:
                            city_dict[cur_city][0] = True
                if (cur_city in city_dict) and city_dict[cur_city][0]:
                    next_city_arr, next_price_arr = city_dict[cur_city][1][0], city_dict[cur_city][1][1]
                    bfs_queue.extend(list(zip(next_city_arr, [next_price + cur_price for next_price in next_price_arr],
                                              [cur_stop + 1 for length in range(len(next_city_arr))])))
                    city_dict[cur_city][0] = False
        return -1 if dst not in price_dict else price_dict[dst]

#543. Diameter of Binary Tree
class DiameterOfBinaryTree(object):
    def diameterOfBinaryTree(self, root):
        def dfs(node):
            yield node
            if node.left:
                yield from dfs(node.left)
                if node.right:
                    yield node
            if node.right:
                yield from dfs(node.right)
            if not node.left and not node.right:
                yield None

        node_arr = list(reversed(list(dfs(root))))
        if len(node_arr) == 2:
            return 0
        cur_length, max_length = 0, 0
        length_dict = {root: 0}
        for prev_node_index, node in enumerate(node_arr[1:]):
            if not node:
                if node_arr[prev_node_index] not in length_dict:
                    length_dict[node_arr[prev_node_index]] = cur_length - 1
                cur_length = 0
                continue
            if node in length_dict:
                max_length = max(max_length, length_dict[node] + cur_length)
                length_dict[node] = max(length_dict[node], cur_length)
                cur_length = length_dict[node]
            cur_length += 1
        return max_length

#100. Same Tree
class SameTree(object):
    def isSameTree(self, p, q):
        def dfs(node):
            yield node.val
            if node.right is not None:
                yield from dfs(node.right)
            else:
                yield 'None_right'
            if node.left is not None:
                yield from dfs(node.left)
            else:
                yield 'None_left'
        
        dfs_res_p = [val for val in dfs(p)]
        dfs_res_q = [val for val in dfs(q)]
        if len(dfs_res_p) != len(dfs_res_q):
            return False
        for num_p, num_q in zip(dfs_res_p, dfs_res_q):
            if num_p != num_q:
                return False
        return True

#513. Find Bottom Left Tree Value
class BottomLeftValue(object):
    def findBottomLeftValue(self, root):
        def dfs(node, lvl):
            if not node.left and not node.right:
                yield (node.val, lvl)
            if node.left:
                yield from dfs(node.left, lvl + 1)
            if node.right:
                yield from dfs(node.right, lvl + 1)
        
        node_arr = [val for val in dfs(root, 0)]
        res_value, max_lvl = root.val, 0
        for value, lvl in node_arr:
            if lvl > max_lvl:
                res_value, max_lvl = value, lvl
        return res_value

#1609. Even Odd Tree
class EvenOddTree(object):
    def isEvenOddTree(self, root):
        def check(cur_val, cur_lvl, prev_val):
            if (cur_lvl % 2) == (cur_val % 2):
                return False
            if not prev_val:
                return True
            else:
                if (cur_val % 2 == 0) and (prev_val > cur_val):
                    return True
                elif ((cur_val % 2 == 1) and (prev_val < cur_val)):
                    return True
                else:
                    return False
                    
        bfs_queue = deque([root])
        cur_lvl, prev_val = 0, None
        while bfs_queue:
            temp_queue = deque([])
            while bfs_queue:
                cur_node = bfs_queue.popleft()
                if not check(cur_node.val, cur_lvl, prev_val):
                    return False
                prev_val = cur_node.val
                if cur_node.left:
                    temp_queue.append(cur_node.left)
                if cur_node.right:
                    temp_queue.append(cur_node.right)
            bfs_queue = temp_queue 
            prev_val = None
            cur_lvl += 1
        return True

class BagOfTokensScore(object):
    def bagOfTokensScore(self, tokens, power):
        cur_score, max_score = 0, 0
        tokens_arr = deque(sorted(tokens))
        while tokens_arr:
            if power < tokens_arr[0]:
                if cur_score > 0:
                    power += tokens_arr.pop()
                    cur_score -= 1
                else:
                    break
            else:
                power -= tokens_arr.popleft()
                cur_score += 1
                max_score = max(max_score, cur_score)
        return max_score

#1750. Minimum Length of String After Deleting Similar Ends
class MinimumLength(object):
    def minimumLength(self, s):
        begin_index = 0
        end_index = len(s) - 1
        while end_index - begin_index >= 1:
            if s[begin_index] != s[end_index]:
                break
            else:
                while (s[begin_index] == s[begin_index+1]) and (begin_index + 1 < end_index):
                    begin_index += 1
                while (s[end_index] == s[end_index-1]) and (end_index - 1 > begin_index):
                    end_index -= 1
                s = s[begin_index+1:end_index]
                begin_index, end_index = 0, len(s) - 1
        return len(s)        

#2485. Find the Pivot Integer
class PivotInteger(object):
    def pivotInteger(self, n):
        pos_sums = {}
        sum = 0
        for num in range(1, n + 1):
            sum += num
            pos_sums[sum] = num
        base_sum = (n + 1) * n / 2
        for num in range(1, n + 1):
            target_sum = (base_sum + num) / 2
            if target_sum in pos_sums:
                return pos_sums[target_sum]
        return -1

#1171. Remove Zero Sum Consecutive Nodes from Linked List
class ZeroSumSublists(object):
    def removeZeroSumSublists(self, head):
        cur_sum = 0
        node_val_arr = []
        sub_sums_arr = [cur_sum]
        cur_node = head
        while cur_node:
            node_val_arr.append(cur_node.val)
            cur_sum += cur_node.val
            sub_sums_arr.append(cur_sum)
            cur_node = cur_node.next
        sub_sums_set = set()
        sub_sums_indexes = {}
        include_indexes = set()
        for index, sum_val in enumerate(sub_sums_arr):
            include_indexes.add(index - 1)
            if sum_val in sub_sums_set:
                start_index = sub_sums_indexes[sum_val]
                sub_sums_set.difference_update(set(sub_sums_arr[start_index:index]))
                include_indexes.difference_update(set(range(start_index, index)))
            sub_sums_indexes[sum_val] = index
            sub_sums_set.add(sum_val)

        res_head = None
        cur_node = None
        for index, val in enumerate(node_val_arr):
            if index in include_indexes:
                if not res_head:
                    res_head = ListNode(val)
                    cur_node = res_head
                else:
                    cur_node.next = ListNode(val)
                    cur_node = cur_node.next
        return res_head

#238. Product of Array Except Self
class ProductExceptSelf(object):
    def productExceptSelf(self, nums):
        prefix, sufix = deque([1]), deque([1])
        for index, val in enumerate(nums[:-1]):
            prefix.append(prefix[index] * val)
        for val in nums[:0:-1]:
            sufix.appendleft(sufix[0] * val)
        res_arr = []
        for index in range(len(nums)):
            res_arr.append(prefix[index] * sufix[index])
        return res_arr

#791. Custom Sort String
class CustomSortString(object):
    def customSortString(self, order, s):
        order_dict = {char: index for index, char in enumerate(order)}
        prefix_arr = [None for index in range(len(order))]
        sufix = ''
        for char in s:
            if char in order_dict:
                if not prefix_arr[order_dict[char]]:
                    prefix_arr[order_dict[char]] = [char, 1]
                else:
                    prefix_arr[order_dict[char]][1] += 1
            else:
                sufix += char
        return ''.join(val[0] * val[1] for val in prefix_arr if val) + sufix

#3005. Count Elements With Maximum Frequency
class MaxFrequencyElements(object):
    def maxFrequencyElements(self, nums):
        num_dict = {}
        max_freq = 0
        num_amount = 0
        for num in nums:
            if num not in num_dict:
                num_dict[num] = 1
            else:
                num_dict[num] += 1
            if num_dict[num] > max_freq:
                max_freq = num_dict[num]
                num_amount = 1
            elif num_dict[num] == max_freq:
                num_amount += 1
        return max_freq * num_amount

#621. Task Scheduler
class LeastInterval(object):
    def leastInterval(self, tasks, n):
        tasks_amount_dict = {}
        tasks_delay_dict = {}
        for task in tasks:
            if task not in tasks_amount_dict:
                tasks_amount_dict[task] = 1
                tasks_delay_dict[task] = 0
            else:
                tasks_amount_dict[task] += 1
        res_length = 0
        while tasks_amount_dict:
            max_task, max_amount = None, -1
            for task in tasks_delay_dict:
                if tasks_delay_dict[task] == 0:
                    if tasks_amount_dict[task] > max_amount:
                        max_task = task
                        max_amount = tasks_amount_dict[task]
                else:
                    tasks_delay_dict[task] += 1
            if max_task:
                tasks_delay_dict[max_task] -= n
                tasks_amount_dict[max_task] -= 1
                if tasks_amount_dict[max_task] == 0:
                    tasks_amount_dict.pop(max_task)
                    tasks_delay_dict.pop(max_task)        
            res_length += 1
        return res_length

#1669. Merge In Between Linked Lists
class MergeInBetween(object):
    def mergeInBetween(self, list1, a, b, list2):
        res_head = ListNode()
        cur_node = res_head
        while list2:
            if a > 0:
                cur_node.val = list1.val
                cur_node.next = ListNode()
                cur_node = cur_node.next
                list1 = list1.next
            elif a <= 0:
                cur_node.val = list2.val
                cur_node.next = ListNode()
                cur_node = cur_node.next
                list2 = list2.next
                if b >= 0:
                    list1 = list1.next
            a -= 1
            b -= 1
        while list1:
            if b >= 0:
                list1 = list1.next
            else:
                cur_node.val = list1.val
                list1 = list1.next
                if list1:
                    cur_node.next = ListNode()
                    cur_node = cur_node.next
            b -= 1
        return res_head

#206. Reverse Linked List
class ReverseList(object):
    def reverseList(self, head):
        def reverse(node):
            if node:
                yield from reverse(node.next)
                yield node.val
        reverse_nodes = reverse(head)
        if not reverse_nodes:
            return None
        res_head = ListNode(next(reverse_nodes))
        cur_node = res_head
        for node in reverse_nodes:
            cur_node.next = ListNode(node)
            cur_node = cur_node.next
        return res_head

#452. Minimum Number of Arrows to Burst Balloons
class MinArrowShots(object):
    def findMinArrowShots(self, points):
        start_end_dict = {}
        for start, end in points:
            if start not in start_end_dict:
                start_end_dict[start] = set()
            start_end_dict[start].add(end)
        end_arr_sorted = deque(sorted([val for value in start_end_dict.values() for val in value]))
        end_amount_dict = {}
        for end in end_arr_sorted:
            if end not in end_amount_dict:
                end_amount_dict[end] = 1
            else:
                end_amount_dict[end] += 1
        start_arr_sorted = deque(sorted(start_end_dict.keys()))
        arrows_amount = 0
        while end_arr_sorted:
            end = end_arr_sorted.popleft()
            if end in end_amount_dict:
                while start_arr_sorted:
                    if start_arr_sorted[0] <= end:
                        start = start_arr_sorted.popleft()
                        if start in start_end_dict:
                            for del_node in start_end_dict[start]:
                                if del_node in end_amount_dict:
                                    end_amount_dict[del_node] -= 1
                                    if end_amount_dict[del_node] == 0:
                                        end_amount_dict.pop(del_node)
                            start_end_dict.pop(start)
                    else: 
                        break
                arrows_amount += 1
        return arrows_amount
        

#234. Palindrome Linked List
class IsPalindrome(object):
    def isPalindrome(self, head):
        def reverse(node, straight_vals, index):
            if node:
                straight_vals.appendleft(node.val)
                yield from reverse(node.next, straight_vals, index + 1)
                if straight_vals[index] != node.val:
                    yield False
        try:
            next(reverse(head, deque([]), 0))
        except:
            return True
        else:
            return False

#442. Find All Duplicates in an Array
class FindDuplicates(object):
    def findDuplicates(self, nums):
        repeat_set = set()
        res_arr = []
        for num in nums:
            if num in repeat_set:
                res_arr.append(num)
            else:
                repeat_set.add(num)
        return res_arr

#930. Binary Subarrays With Sum
class NumSubarraysWithSum(object):
    def numSubarraysWithSum(self, nums, goal):
        left_pointer, right_pointer = 0, 0
        ones_amount = nums[right_pointer]
        total_amount = 0
        while right_pointer < len(nums):
            if ones_amount < goal:
                right_pointer += 1
                if right_pointer < len(nums):
                    ones_amount += nums[right_pointer]
            elif ones_amount == goal:
                zero_left, zero_right = 0, 0
                while (right_pointer < len(nums) - 1) and (nums[right_pointer+1] == 0):
                    right_pointer += 1
                    zero_right += 1
                left_index = left_pointer
                while (left_index <= right_pointer) and (not nums[left_index]):
                    zero_left += 1 - nums[left_index]
                    left_index += 1
                if not goal:
                    length = right_pointer - left_pointer + 1
                    total_amount += length * (length + 1) // 2
                else:
                    total_amount += (zero_left + 1) * (zero_right + 1)
                right_pointer += 1
                if right_pointer < len(nums):
                    ones_amount += nums[right_pointer]
            else:
                ones_amount -= nums[left_pointer]
                left_pointer += 1
        return total_amount
            

#41. First Missing Positive
class FirstMissingPositive(object):
    def firstMissingPositive(self, nums):
        nums_set = set(nums)
        first_pos_num = 1
        while True:
            if first_pos_num not in nums_set:
                return first_pos_num
            first_pos_num += 1

#9. Palindrome Number
class IsPalindrome(object):
    def isPalindrome(self, x):
        string_x = str(abs(x))
        for index in range((len(string_x) + 1) // 2):
            if string_x[index] != string_x[-(index+1)]:
                return False
        return True

#2958. Length of Longest Subarray With at Most K Frequency
class maxSubarrayLength(object):
    def maxSubarrayLength(self, nums, k):
        freq_dict = {}
        max_freq = 0
        left_pointer, right_pointer = 0, 0
        max_length = 0
        for right_pointer in range(len(nums)):
            if nums[right_pointer] not in freq_dict:
                freq_dict[nums[right_pointer]] = 1
            else:
                freq_dict[nums[right_pointer]] += 1
            max_freq = max(max_freq, freq_dict[nums[right_pointer]])
            if max_freq > k:
                max_length = max(max_length, right_pointer - left_pointer)
                break_flag = False
                while not break_flag:
                    if freq_dict[nums[left_pointer]] == max_freq:
                        max_freq -= 1
                        break_flag = True    
                    freq_dict[nums[left_pointer]] -= 1
                    left_pointer += 1
        max_length = max(max_length, right_pointer - left_pointer + 1)
        return max_length

#58. Length of Last Word
class LengthOfLastWord(object):
    def lengthOfLastWord(self, s):
        return len(s.split()[-1])

#713. Subarray Product Less Than K
class NumSubarrayProductLessThanK:
    def numSubarrayProductLessThanK(self, nums, k):
        def countVariants(length):
            return length * (length + 1) // 2
        def addAmount(left_pointer, right_pointer, last_index):
            res_amount = countVariants(right_pointer - left_pointer)
            if last_index > left_pointer:
                res_amount -= countVariants(last_index - left_pointer)
            return res_amount
        
        left_pointer, right_pointer = 0, 0
        total_amount = 0
        cur_product = 1
        last_index = -1
        while right_pointer <= len(nums) - 1:
            cur_product *= nums[right_pointer]
            if cur_product >= k:
                total_amount += addAmount(left_pointer, right_pointer, last_index)
                last_index = right_pointer
                while (left_pointer <= right_pointer) and (cur_product >= k):
                    cur_product /= nums[left_pointer]
                    left_pointer += 1
            right_pointer += 1
        total_amount += addAmount(left_pointer, right_pointer, last_index)
        return total_amount

#205. Isomorphic Strings
class IsIsomorphic(object):
    def isIsomorphic(self, s, t):
        change_dict = {}
        if len(s) != len(t):
            return False
        for index in range(len(s)):
            if s[index] not in change_dict:
                change_dict[s[index]] = t[index]
            else:
                if change_dict[s[index]] != t[index]:
                    return False
        values_arr = change_dict.values()
        if len(values_arr) != len(set(values_arr)):
            return False
        return True

#2962. Count Subarrays Where Max Element Appears at Least K Times
class CountSubarrays(object):
    def countSubarrays(self, nums, k):
        max_num = max(nums)
        max_num_margin = []
        left_margin = 0
        for num in nums:
            if num == max_num:
                if len(max_num_margin) > 0:
                    max_num_margin[-1].append(left_margin)
                max_num_margin.append([left_margin])
                left_margin = 0
                continue
            left_margin += 1
        max_num_margin[-1].append(left_margin)

        total_amount = 0
        left_margin = 0
        left_margin_index = 0
        for index in range(k - 1, len(max_num_margin)):
            left_margin += max_num_margin[left_margin_index][0] + 1
            total_amount += left_margin * (max_num_margin[index][1] + 1)
            left_margin_index += 1
        return total_amount       

#1614. Maximum Nesting Depth of the Parentheses
class MaxDepth(object):
    def maxDepth(self, s):
        cur_depth, max_depth = 0, 0
        for char in s:
            if char == '(':
                cur_depth += 1
                max_depth = max(max_depth, cur_depth)
            elif char == ')':
                cur_depth -= 1
        return max_depth

#1544. Make The String Great
class MakeGood(object):
    def makeGood(self, s):
        end_flag = False
        while (not end_flag) and s:
            end_flag = True
            for index in range(1, len(s)):
                if (s[index-1] != s[index]) and ((s[index-1].upper() == s[index]) or (s[index-1].lower() == s[index])):
                    if len(s) == 2:
                        s = ''
                    else:
                        s = s[:index-1] + s[index+1:]
                    end_flag = False
                    break 
        return s    

#25. Reverse Nodes in k-Group
class ReverseKGroup(object):
    def reverseKGroup(self, head, k):
        def generateList(value_arr):
            head = ListNode(value_arr[0])
            cur_val = head
            for val in value_arr[1:]:
                cur_val.next = ListNode(val)
                cur_val = cur_val.next
            return head
            
        iter = k
        reverse_list = []
        iter_list = deque([])
        while head:
            if not iter:
                reverse_list.extend(iter_list)
                iter = k
            if iter == k:
                iter_list = deque([])
            iter_list.appendleft(head.val)
            head = head.next
            iter -= 1
        for index in range(1, len(iter_list) + 1):
            if len(iter_list) == k:
                reverse_list.append(iter_list[index-1])
            else:
                reverse_list.append(iter_list[-index])
        return generateList(reverse_list)

#1700. Number of Students Unable to Eat Lunch
class CountStudents(object):
    def countStudents(self, students, sandwiches):
        students = deque(students)
        sandwiches = deque(sandwiches)
        while True:
            iter = len(students)
            cycle_flag = True
            while iter:
                cur_student = students.popleft()
                if cur_student == sandwiches[0]:
                    sandwiches.popleft()
                    cycle_flag = False
                else:
                    students.append(cur_student)
                iter -= 1
            if cycle_flag or not len(students):
                return len(students)

#79. Word Search
class Exist(object):
    def exist(self, board, word):
        def nextCoord(cur_coord, cur_route, next_letter, board):
            cur_row, cur_col = cur_coord[0], cur_coord[1]
            if ((cur_col + 1 < len(board[cur_row])) 
                and (board[cur_row][cur_col+1] == next_letter) 
                and (cur_row, cur_col + 1) not in cur_route):
                yield (cur_row, cur_col + 1)
            if ((cur_col - 1 >= 0) 
                and (board[cur_row][cur_col-1] == next_letter)
                and (cur_row, cur_col - 1) not in cur_route):
                yield (cur_row, cur_col - 1)
            if ((cur_row + 1 < len(board)) 
                and (board[cur_row+1][cur_col] == next_letter)
                and (cur_row + 1, cur_col) not in cur_route):
                yield (cur_row + 1, cur_col)
            if ((cur_row - 1 >= 0) 
                and (board[cur_row-1][cur_col] == next_letter)
                and (cur_row - 1, cur_col) not in cur_route):
                yield (cur_row - 1, cur_col)        

        for row_coord in range(len(board)):
            for col_coord in range(len(board[row_coord])):
                if board[row_coord][col_coord] == word[0]:
                    checked_coord = set()
                    cur_coord = (row_coord, col_coord)
                    dfs_stack = deque([(0, cur_coord, 0)])
                    cur_route, route_dict = set(), dict()
                    cur_route.add(cur_coord)
                    route_dict[0] = [cur_route]
                    while dfs_stack:
                        cur_letter_index, cur_coord, cur_route_index = dfs_stack.pop()
                        if cur_letter_index == len(word) - 1:
                            return True
                        cur_route = route_dict[cur_letter_index][cur_route_index]
                        if (cur_letter_index + 1) not in route_dict:
                            route_dict[cur_letter_index + 1] = []
                        for next_coord in nextCoord(cur_coord, cur_route, word[cur_letter_index+1], board):
                            checked_coord.add(next_coord)
                            dfs_stack.append((cur_letter_index + 1, next_coord, len(route_dict[cur_letter_index + 1])))
                            next_route = set(cur_route)
                            next_route.add(next_coord)
                            route_dict[cur_letter_index + 1].append(next_route)  
                    if len(checked_coord) == len(board) * len(board[0]):
                        return False
        return False

#2073. Time Needed to Buy Tickets
class TimeRequiredToBuy(object):
    def timeRequiredToBuy(self, tickets, k):
        time_sec = 0
        while tickets[k] > 0:
            for index in range(len(tickets)):
                if tickets[index] > 0:
                    tickets[index] -= 1
                    time_sec += 1
        return time_sec

#129. Sum Root to Leaf Numbers
class SumNumberssumNumbers(object):
    def sumNumbers(self, root):
        dfs_stack = deque([(root, root.val)])
        full_sum = 0
        while dfs_stack:
            cur_node, cur_num = dfs_stack.pop()
            if not cur_node.left and not cur_node.right:
                full_sum += cur_num
                continue
            if cur_node.left:
                dfs_stack.append((cur_node.left, cur_num * 10 + cur_node.left.val))
            if cur_node.right:
                dfs_stack.append((cur_node.right, cur_num * 10 + cur_node.right.val))
        return full_sum

#623. Add One Row to Tree
class addOneRow(object):
    def addOneRow(self, root, val, depth):
        if depth == 1:
            return TreeNode(val, root)
        dfs_stack = deque([(root, 1)])
        while dfs_stack:
            cur_node, cur_depth = dfs_stack.pop()
            if cur_depth < depth - 1:
                if cur_node.left:
                    dfs_stack.append((cur_node.left, cur_depth + 1))
                if cur_node.right:
                    dfs_stack.append((cur_node.right, cur_depth + 1))
            elif cur_depth == depth - 1:
                left_node = TreeNode(val, left=cur_node.left)
                cur_node.left = left_node
                right_node = TreeNode(val, right=cur_node.right)
                cur_node.right = right_node
        return root

#988. Smallest String Starting From Leaf
class StringLetterLess(Exception):
    pass

class StringLetterGreater(Exception):
    pass

class SmallestFromLeaf(object):
    def smallestFromLeaf(self, root):
        def builtRoutes(node, arr, lvl):
            if node:
                if lvl:
                    arr = list(arr[:lvl])
                arr.append(node.val)
                if not node.left and not node.right:
                    yield list(reversed(arr))
                yield from builtRoutes(node.left, arr, lvl+1)
                yield from builtRoutes(node.right, arr, lvl+1)                

        string_list = list(builtRoutes(root, list(), 0))
        min_string = string_list[0]
        for string in string_list[1:]:
            if string[0] < min_string[0]:
                min_string = string
            elif string[0] == min_string[0]:
                try:
                    for index in range(1, min(len(min_string), len(string))):
                        if string[index] < min_string[index]:
                            raise StringLetterLess()
                        elif string[index] > min_string[index]:
                            raise StringLetterGreater()
                    if len(string) < len(min_string):
                        min_string = string
                except StringLetterLess:
                    min_string = string
                except StringLetterGreater:
                    continue
        return ''.join(chr(code + 97) for code in min_string)

#463. Island Perimeter
class IslandPerimeter(object):
    def islandPerimeter(self, grid):
        def amountOfWaterSides(index_row, index_col, grid):
            sides = 0
            if (index_col - 1 < 0) or ((index_col - 1 >= 0) and (grid[index_row][index_col-1] == 0)):
                sides += 1
            if (index_col + 1 == len(grid[index_row])) or ((index_col + 1 <= len(grid[index_row]) - 1) 
                                                           and (grid[index_row][index_col+1] == 0)):
                sides += 1    
            if (index_row - 1 < 0) or ((index_row - 1 >= 0) and (grid[index_row-1][index_col] == 0)):
                sides += 1
            if (index_row + 1 == len(grid)) or ((index_row + 1 <= len(grid) - 1) 
                                                and (grid[index_row+1][index_col] == 0)):
                sides += 1
            return sides
            
        perim = 0
        for index_row in range(len(grid)):
            for index_col in range(len(grid[index_row])):
                if grid[index_row][index_col] == 1:
                    perim += amountOfWaterSides(index_row, index_col, grid)
        return perim

#200. Number of Islands
class NumIslands(object):
    def numIslands(self, grid):
        islands_amount = 0
        for index_row in range(len(grid)):
            for index_col in range(len(grid[index_row])):
                if grid[index_row][index_col] == '1':
                    dfs_stack = deque([(index_row, index_col)])
                    while dfs_stack:
                        cur_row, cur_col = dfs_stack.pop()
                        grid[cur_row][cur_col] = '0'
                        if ((cur_col + 1 <= len(grid[cur_row]) - 1) 
                             and (grid[cur_row][cur_col+1] == '1')):
                            dfs_stack.append((cur_row, cur_col + 1))
                        if ((cur_row + 1 <= len(grid) - 1) 
                             and (grid[cur_row+1][cur_col] == '1')):
                            dfs_stack.append((cur_row + 1, cur_col))
                        if ((cur_row - 1 >= 0) 
                             and (grid[cur_row-1][cur_col] == '1')):
                            dfs_stack.append((cur_row - 1, cur_col))
                        if ((cur_col - 1 >= 0) 
                             and (grid[cur_row][cur_col-1] == '1')):
                            dfs_stack.append((cur_row, cur_col - 1))
                    islands_amount += 1
        return islands_amount

#37. Sudoku Solver
class EmptyCell(Exception):
    pass

class SolveSudoku(object):
    def solveSudoku(self, board):
        def solve(board, row_dict, col_dict, sub_matrix_dict, index_row, index_col):  
            if (index_row, index_col) == (9, 0):
                try:
                    for row in board:
                        for elem in row:
                            if elem == '.':
                                raise EmptyCell()
                    yield board
                except EmptyCell:
                    pass
            elif board[index_row][index_col] == '.':
                full_set = set(range(1, 10))
                cur_cell = (full_set 
                            - (set() if index_row not in row_dict else row_dict[index_row]) 
                            - (set() if index_col not in col_dict else col_dict[index_col] )
                            - (set() if (index_row // 3, index_col // 3) not in sub_matrix_dict
                                     else sub_matrix_dict[(index_row // 3, index_col // 3)]))
                for new_num in cur_cell:
                    new_board = [list(row) for row in board]
                    new_board[index_row][index_col] = str(new_num)
                    
                    new_row_dict = {key: set(row_dict[key]) for key in row_dict.keys()}
                    if index_row not in new_row_dict:
                        new_row_dict[index_row] = set()
                    new_row_dict[index_row].add(new_num)

                    new_col_dict = {key: set(col_dict[key]) for key in col_dict.keys()}
                    if index_col not in new_col_dict:
                        new_col_dict[index_col] = set()
                    new_col_dict[index_col].add(new_num)

                    new_sub_matrix_dict = {key: set(sub_matrix_dict[key]) for key in sub_matrix_dict.keys()}
                    coord = (index_row // 3, index_col // 3)
                    if coord not in new_sub_matrix_dict:
                        new_sub_matrix_dict[coord] = set()
                    new_sub_matrix_dict[coord].add(new_num)
                    
                    new_index_row = index_row
                    new_index_col = index_col
                    while new_board[new_index_row][new_index_col] != '.':
                        new_index_col += 1
                        if new_index_col == 9:
                            new_index_col = 0
                            new_index_row += 1
                        if (new_index_row, new_index_col) == (9, 0):
                            break    
                    yield from solve(new_board, new_row_dict, new_col_dict, new_sub_matrix_dict, new_index_row, new_index_col)
           
        row_dict, col_dict, sub_matrix_dict = {}, {}, {}
        first_index_row, first_index_col = -1, -1
        for index_row in range(len(board)):
            row_dict[index_row] = set()
            for index_col in range(len(board[index_row])):
                if board[index_row][index_col] != '.':
                    row_dict[index_row].add(int(board[index_row][index_col]))
                    if index_col not in col_dict:
                        col_dict[index_col] = set()
                    col_dict[index_col].add(int(board[index_row][index_col]))
                    sub_matrix_index = (index_row // 3, index_col // 3)
                    if sub_matrix_index not in sub_matrix_dict:
                        sub_matrix_dict[sub_matrix_index] = set()
                    sub_matrix_dict[sub_matrix_index].add(int(board[index_row][index_col]))
                else:
                    if first_index_row < 0 and first_index_col < 0:
                        first_index_row, first_index_col = index_row, index_col
        
        res_board = list(solve(board, row_dict, col_dict, sub_matrix_dict, first_index_row, first_index_col))[0]
        for index_row in range(len(board)):
            for index_col in range(len(board[index_row])):
                board[index_row][index_col] = res_board[index_row][index_col]

#752. Open the Lock
class OpenLock(object):    
    def openLock(self, deadends, target):
        def buildNewLock(new_num, num_index, cur_lock):
            cur_lock_arr = list(cur_lock)
            cur_lock_arr[num_index] = new_num
            return ''.join(cur_lock_arr)

        def checkedEdge(cur_edge, checked_edges):
            return cur_edge in checked_edges or tuple(reversed(cur_edge)) in checked_edges 
        
        checked_edges = set()
        deadens_set = set(deadends)
        bfs_queue = deque([])
        start_lock = '0000'
        if start_lock not in deadens_set:
            bfs_queue.append(start_lock)
        min_turns = 0
        while bfs_queue:
            temp_queue = deque([])
            while bfs_queue:
                cur_lock = bfs_queue.popleft()
                if cur_lock == target:
                    return min_turns
                for num_index in range(len(cur_lock)):
                    upper_num = str((int(cur_lock[num_index]) + 1) % 10)
                    upper_lock = buildNewLock(upper_num, num_index, cur_lock)
                    upper_edge = (cur_lock, upper_lock)
                    if upper_lock not in deadens_set and not checkedEdge(upper_edge, checked_edges):
                        temp_queue.append(upper_lock)
                        checked_edges.add(upper_edge)

                    lower_num = str((int(cur_lock[num_index]) + 9) % 10)
                    lower_lock = buildNewLock(lower_num, num_index, cur_lock)
                    lower_edge = (cur_lock, lower_lock)
                    if lower_lock not in deadens_set and not checkedEdge(lower_edge, checked_edges):
                        temp_queue.append(lower_lock)
                        checked_edges.add(lower_edge)
            bfs_queue = temp_queue
            min_turns += 1
        return -1            

#1992. Find All Groups of Farmland
class FindFarmland(object):
    def findFarmland(self, land):
        def updateBottomRight(bottom_row, right_col, cur_row, cur_col):
            if (cur_row > bottom_row) or ((cur_row == bottom_row) and (cur_col > right_col)):
                return cur_row, cur_col
            else:
                return bottom_row, right_col
        
        farmland_coord_arr = []
        for index_row in range(len(land)):
            for index_col in range(len(land[index_row])):
                if land[index_row][index_col] == 1:
                    dfs_stack = deque([(index_row, index_col)])
                    farmland_coord = [index_row, index_col]
                    bottom_row, right_col = index_row, index_col
                    while dfs_stack:
                        cur_row, cur_col = dfs_stack.pop()
                        land[cur_row][cur_col] = 0
                        bottom_row, right_col = updateBottomRight(bottom_row, right_col, cur_row, cur_col)
                        if (cur_row - 1 >= 0) and (land[cur_row-1][cur_col] == 1):
                            dfs_stack.append((cur_row - 1, cur_col))
                        if (cur_row + 1 < len(land)) and (land[cur_row+1][cur_col] == 1):
                            dfs_stack.append((cur_row + 1, cur_col))
                        if (cur_col - 1 >= 0) and (land[cur_row][cur_col-1] == 1):
                            dfs_stack.append((cur_row, cur_col - 1))
                        if (cur_col + 1 < len(land[index_row])) and (land[cur_row][cur_col+1] == 1):
                            dfs_stack.append((cur_row, cur_col + 1))
                    farmland_coord.extend([bottom_row, right_col])
                    farmland_coord_arr.append(farmland_coord)
        return farmland_coord_arr

#1971. Find if Path Exists in Graph
class ValidPath(object):
    def validPath(self, n, edges, source, destination):
        neighbours_dict = {}
        for edge in edges:
            begin, end = edge[0], edge[1]
            if begin not in neighbours_dict:
                neighbours_dict[begin] = set()
            neighbours_dict[begin].add(end)
            if end not in neighbours_dict:
                neighbours_dict[end] = set()
            neighbours_dict[end].add(begin)

        dfs_stack = deque([source])
        checked_nodes = set()
        while dfs_stack:
            cur_node = dfs_stack.pop()
            if cur_node == destination:
                return True
            checked_nodes.add(cur_node)
            pos_neighbours = neighbours_dict[cur_node] - checked_nodes
            dfs_stack.extend(pos_neighbours)
        return False

#1137. N-th Tribonacci Number
class Tribonacci(object):
    def tribonacci(self, n):
        tribonacci_arr = [0, 1, 1]
        while len(tribonacci_arr) <= n:
            next_num = tribonacci_arr[-1] + tribonacci_arr[-2] + tribonacci_arr[-3]
            tribonacci_arr.append(next_num)
        return tribonacci_arr[n]

#1289. Minimum Falling Path Sum II
class MinFallingPathSum(object):
    def minFallingPathSum(self, grid):
        if len(grid) == 1:
            return grid[0][0]
        first_min_index = 0 if grid[0][0] < grid[0][1] else 1
        second_min_index = 0 if grid[0][0] >= grid[0][1] else 1
        for row_index in range(len(grid)):
            new_first_min_index, new_second_min_index = 0, 0
            for col_index in range(len(grid)):
                if (row_index == 0) and (col_index >= 2):
                    if grid[row_index][col_index] < grid[row_index][first_min_index]:
                        second_min_index = first_min_index
                        first_min_index = col_index
                    elif grid[row_index][col_index] < grid[row_index][second_min_index]:
                        second_min_index = col_index
                elif row_index != 0:
                    if col_index != first_min_index:
                        grid[row_index][col_index] += grid[row_index-1][first_min_index]
                    else:
                        grid[row_index][col_index] += grid[row_index-1][second_min_index]
                        
                    if col_index >= 2:
                        if grid[row_index][col_index] < grid[row_index][new_first_min_index]:
                            new_second_min_index = new_first_min_index
                            new_first_min_index = col_index
                        elif grid[row_index][col_index] < grid[row_index][new_second_min_index]:
                            new_second_min_index = col_index
                    elif col_index == 1:
                        new_first_min_index = 0 if grid[row_index][0] < grid[row_index][1] else 1
                        new_second_min_index = 0 if grid[row_index][0] >= grid[row_index][1] else 1
            if row_index != 0:
                first_min_index, second_min_index = new_first_min_index, new_second_min_index
        return min(grid[-1])

#2370. Longest Ideal Subsequence
class LongestIdealString(object):
    def longestIdealString(self, s, k):
        subseq_dict = {}
        for char in s:
            for dif in range(k + 1):
                if dif == 0:
                    if char not in subseq_dict:
                        subseq_dict[char] = 1
                    else:
                        subseq_dict[char] += 1
                else:
                    upper_char = chr(ord(char) + dif) if ord(char) + dif <= ord('z') else None
                    lower_char = chr(ord(char) - dif) if ord(char) - dif >= ord('a') else None
                    if upper_char and upper_char in subseq_dict:
                        subseq_dict[char] = max(subseq_dict[char], subseq_dict[upper_char] + 1)
                    if lower_char and lower_char in subseq_dict:
                        subseq_dict[char] = max(subseq_dict[char], subseq_dict[lower_char] + 1)                        
        return max(subseq_dict.values())

#1863. Sum of All Subset XOR Totals
class SubsetXORSum(object):
    def subsetXORSum(self, nums):
        cur_subsets_arr = []
        res_sum = 0
        for index, num in enumerate(nums):
            index_seq = deque([index])
            cur_subsets_arr.append((num, index_seq))
            res_sum += num
        while cur_subsets_arr:
            new_subsets_arr = []
            for cur_subsets in cur_subsets_arr:
                cur_num, cur_subset = cur_subsets
                for index in range(cur_subset[-1] + 1, len(nums)):
                    new_subset = list(cur_subset)
                    new_subset.append(index)
                    new_num = cur_num ^ nums[index]
                    new_subsets_arr.append((new_num, new_subset))
                    res_sum += new_num
            cur_subsets_arr = new_subsets_arr
        return res_sum
        

#78. Subsets
class Subsets(object):
    def subsets(self, nums):
        result = [[]]
        cur_elem_arr = [([num], index) for index, num in enumerate(nums)]
        while cur_elem_arr:
            new_elem_arr = []
            for cur_elem in cur_elem_arr:
                cur_arr, cur_index = cur_elem
                result.append(cur_arr)
                for index in range(cur_index + 1, len(nums)):
                    new_arr = list(cur_arr)
                    new_arr.append(nums[index])
                    new_elem_arr.append((new_arr, index))
            cur_elem_arr = new_elem_arr
        return result

#131. Palindrome Partitioning
class Partition(object):
    def partition(self, s):
        partition_dict = {}
        cur_partition_arr = [(string, index) for index, string in enumerate(s)]
        while cur_partition_arr:
            new_partition_arr = []
            for cur_partition in cur_partition_arr:
                cur_string, cur_index = cur_partition
                if cur_string == ''.join(reversed(cur_string)):
                    cur_start = cur_index + 1 - len(cur_string)
                    cur_end = cur_index 
                    if cur_start not in partition_dict:
                        partition_dict[cur_start] = dict()
                    partition_dict[cur_start][cur_end] = cur_string
                if cur_index + 1 < len(s):
                    new_partition_arr.append((cur_string + s[cur_index+1], cur_index + 1))
            cur_partition_arr = new_partition_arr

        result = []
        for first_key in partition_dict[0].keys():
            first_string = [partition_dict[0][first_key]]
            first_end = first_key
            cur_result_arr = [(first_string, first_end)]
            while cur_result_arr:
                new_result_arr = []
                for cur_result in cur_result_arr:
                    cur_string, cur_end = cur_result
                    if cur_end + 1 in partition_dict:
                        for new_end in partition_dict[cur_end+1].keys():
                            new_string = list(cur_string)
                            new_string.append(partition_dict[cur_end+1][new_end])
                            new_result_arr.append((new_string, new_end))
                    else:
                        result.append(cur_string)
                cur_result_arr = new_result_arr
        return result

#2597. The Number of Beautiful Subsets
class BeautifulSubsets(object):
    def beautifulSubsets(self, nums, k):
        exception_dict = {}
        for main_index in range(len(nums)):
            for except_index in range(main_index + 1, len(nums)):
                if abs(nums[main_index] - nums[except_index]) == k:
                    if main_index not in exception_dict:
                        exception_dict[main_index] = set()
                    exception_dict[main_index].add(except_index)
                    if except_index not in exception_dict:
                        exception_dict[except_index] = set()
                    exception_dict[except_index].add(main_index)

        res_amount = 0
        cur_subset_index_arr = [({index}, index) for index in range(len(nums))]
        while cur_subset_index_arr:
            new_subset_index_arr = []
            for cur_subset_index in cur_subset_index_arr:
                cur_subset, cur_index = cur_subset_index
                res_amount += 1
                for new_index in range(cur_index + 1, len(nums)):
                    if (new_index not in exception_dict) or (not (exception_dict[new_index] & cur_subset)):
                        new_subset = set(cur_subset)
                        new_subset.add(new_index)
                        new_subset_index_arr.append((new_subset, new_index))
            cur_subset_index_arr = new_subset_index_arr                        
        return res_amount

#1442. Count Triplets That Can Form Two Arrays of Equal XOR
class CountTriplets(object):
    def countTriplets(self, arr):
        def xor_res(index_tuple):
            if index_tuple[0] == index_tuple[1]:
                return arr[index_tuple[0]]
            else:
                return xor_res_dict[(index_tuple[0], index_tuple[1] - 1)] ^ arr[index_tuple[1]]
        
        if len(arr) == 1:
            return 0
        res_amount = 0
        xor_res_dict = {}
        cur_indexes = [0, 1, 1]
        while True:
            a_tuple, b_tuple = (cur_indexes[0], cur_indexes[1] - 1), (cur_indexes[1], cur_indexes[2])
            if a_tuple not in xor_res_dict:
                xor_res_dict[a_tuple] = xor_res(a_tuple)
            if b_tuple not in xor_res_dict:
                xor_res_dict[b_tuple] = xor_res(b_tuple)
            if xor_res_dict[a_tuple] == xor_res_dict[b_tuple]:
                res_amount += 1
            cur_indexes[2] += 1
            if cur_indexes[2] == len(arr):
                cur_indexes[1] += 1
                cur_indexes[2] = cur_indexes[1]
            if cur_indexes[2] == len(arr):
                cur_indexes[0] += 1
                cur_indexes[1] = cur_indexes[0] + 1
                cur_indexes[2] = cur_indexes[1]
            if cur_indexes[2] == len(arr):
                return res_amount

#260. Single Number III
class SingleNumber(object):
    def singleNumber(self, nums):
        xor_result = functools.reduce(lambda x, y: x ^ y, nums)
        dif_digit_index = bin(xor_result)[::-1].find('1')
        group_with_0, group_with_1 = 0, 0
        for num in nums:
            if bin(num >> dif_digit_index)[-1] == '0':
                group_with_0 ^= num
            else:
                group_with_1 ^= num
        return [group_with_0, group_with_1]

#1404. Number of Steps to Reduce a Number in Binary Representation to One
class NumSteps(object):
    def numSteps(self, s):
        res_amount = 0
        while len(s) > 1:
            if s[-1] == '0':
                s = s[:-1]
            else:
                s_arr_rev = list(reversed(s))
                cur_index = 0
                while cur_index < len(s):
                    if s_arr_rev[cur_index] == '1':
                        s_arr_rev[cur_index] = '0'
                    else:
                        s_arr_rev[cur_index] = '1'
                        break
                    cur_index += 1
                if cur_index == len(s):
                    s = '1' + '0' * (cur_index)
                else:
                    s = ''.join(reversed(s_arr_rev))
            res_amount += 1
        return res_amount

#1382. Balance a Binary Search Tree
class BalanceBST(object):
    def balanceBST(self, root):
        def treeTraversal(root):
            yield root.val
            if root.left:
                yield from treeTraversal(root.left)
            if root.right:
                yield from treeTraversal(root.right)
                
        def buildTree(root, nodes_val):
            root.val = nodes_val[len(nodes_val)//2]
            if len(nodes_val) > 2:
                root.right = TreeNode()
                buildTree(root.right, nodes_val[(len(nodes_val)//2+1):])
            if len(nodes_val) > 1:
                root.left = TreeNode()
                buildTree(root.left, nodes_val[:(len(nodes_val)//2)])        
        
        nodes_val = list(sorted(treeTraversal(root)))
        root = TreeNode()
        buildTree(root, nodes_val)
        return root

#2285. Maximum Total Importance of Roads
class MaximumImportance(object):
    def maximumImportance(self, n, roads):
        city_degree = {}
        for road in roads:
            for city in road:
                if city not in city_degree:
                    city_degree[city] = 1
                else:
                    city_degree[city] += 1
        city_importance = {}
        importance = n
        for city in sorted(city_degree.keys(), key=lambda x: city_degree[x], reverse=True):
            city_importance[city] = importance
            importance -= 1
        result = 0
        for road in roads:
            result += city_importance[road[0]] + city_importance[road[1]]
        return result

#1823. Find the Winner of the Circular Game
class FindTheWinner(object):
    def findTheWinner(self, n, k):
        cur_arr = list(range(1, n + 1))
        player_set = set(cur_arr)
        cur_index, counter = 0, k
        while len(player_set) > 1:
            counter -= 1
            if not cur_arr[cur_index]:
                    counter += 1
            if not counter:
                player_set.remove(cur_arr[cur_index])
                cur_arr[cur_index] = 0
                counter = k
            cur_index += 1
            if cur_index == len(cur_arr):
                cur_index = 0
        return player_set.pop()

#51. N-Queens
class SolveNQueens(object):
    def solveNQueens(self, n):
        def buildNextRowPattern(cur_row):
            next_row_pattern = [(0, []) for count in range(n)]
            for cur_index, cur_elem in enumerate(cur_row):
                status_arr = cur_elem[1]
                for status in status_arr:
                    next_index = cur_index + status
                    if 0 <= next_index < n:
                        next_row_pattern[next_index][1].append(status)
            return next_row_pattern 

        def convertInResMatrix(origin_matrix):
            res_matrix = []
            for row in origin_matrix:
                res_row_arr = []
                for elem in row:
                    if not elem[0]:
                        res_row_arr.append('.')
                    else:
                        res_row_arr.append('Q')
                res_matrix.append(''.join(res_row_arr))
            return res_matrix
            
        cur_matrix_arr = [[[(0, []) if col_index != count else (1, [-1, 0, 1]) 
                       for col_index in range(n)]] for count in range(n)]
        for row_index in range(n - 1):
            temp_matrix_arr = []
            for cur_matrix in cur_matrix_arr:
                next_row_pattern = buildNextRowPattern(cur_matrix[-1])
                for index, elem in enumerate(next_row_pattern):
                    if not elem[1]:
                        next_row = list(next_row_pattern)
                        next_row[index] = (1, [-1, 0, 1])
                        next_matrix = [list(cur_row) for cur_row in cur_matrix]
                        next_matrix.append(next_row)
                        temp_matrix_arr.append(next_matrix)
            cur_matrix_arr = temp_matrix_arr    

        result = []
        for matrix in cur_matrix_arr:
            result.append(convertInResMatrix(matrix))
        return result

#1518. Water Bottles
class NumWaterBottles(object):
    def numWaterBottles(self, numBottles, numExchange):
        def exchangeNum(numEmpty, numExchange):
            if numEmpty >= numExchange:
                numFull = numEmpty // numExchange
                yield numFull
                yield from exchangeNum(numFull + numEmpty % numExchange, numExchange)
        return numBottles + sum(exchangeNum(numBottles, numExchange))

#2058. Find the Minimum and Maximum Number of Nodes Between Critical Points
class NodesBetweenCriticalPoints(object):
    def nodesBetweenCriticalPoints(self, head):
        critPoints_arr = []
        prev = None
        index = 0
        while head:
            if head.next:
                if prev and head.next:
                     if (prev.val < head.val > head.next.val) or (prev.val > head.val < head.next.val):
                         critPoints_arr.append(index)
            prev = head
            head = head.next
            index += 1
        if len(critPoints_arr) < 2:
            return [-1, -1]
        else:
            max_dist = critPoints_arr[-1] - critPoints_arr[0]
            min_dist = max_dist
            prev_index = critPoints_arr[0] 
            for index in critPoints_arr[1:]:
                min_dist = min(min_dist, index - prev_index)
                prev_index = index
            return [min_dist, max_dist]

#2582. Pass the Pillow
class PassThePillow(object):
    def passThePillow(self, n, time):
        circle_amount = time // (n - 1)
        last_bias = time % (n - 1)
        if circle_amount % 2:
            return n - last_bias
        else:
            return 1 + last_bias

#1701. Average Waiting Time
class AverageWaitingTime(object):
    def averageWaitingTime(self, customers):
        full_waiting_time = customers[0][1]
        cur_time = sum(customers[0])
        for customer in customers[1:]:
            if cur_time - customer[0] < 0:
                full_waiting_time += customer[1]
                cur_time = sum(customer)
            else:
                cur_time += customer[1]
                full_waiting_time += cur_time - customer[0]
        return full_waiting_time / len(customers)

#2181. Merge Nodes in Between Zeros
class MergeNodes(object):
    def mergeNodes(self, head):
        head_flag = True
        res_head, cur_node = ListNode(), None
        cur_val = 0
        head = head.next
        while head:
            if not head.val:
                if not head_flag:
                    cur_node.next = ListNode(cur_val)
                    cur_node = cur_node.next
                else:
                    res_head.val = cur_val
                    cur_node = res_head
                    head_flag = False
                cur_val = 0
            else:
                cur_val += head.val
            head = head.next
        return res_head

#1598. Crawler Log Folder
class MinOperations(object):
    def minOperations(self, logs):
        deep = 0
        for log in logs:
            deep += 1 - len(re.findall('[.]', log))
            if deep < 0:
                deep = 0
        return deep

#1509. Minimum Difference Between Largest and Smallest Value in Three Moves
class MinDifference(object):
    def minDifference(self, nums):
        if len(nums) <= 4:
            return 0
        nums.sort()
        min_diff = nums[-1] - nums[0]
        for counter in range(4):
            min_diff = min(min_diff, nums[-(4-counter)] - nums[counter])
        return min_diff

#2196. Create Binary Tree From Descriptions
class CreateBinaryTree(object):
    def createBinaryTree(self, descriptions):
        def createParent(parent_val, val_node_dict):
            parent = TreeNode(parent_val)
            val_node_dict[parent_val] = parent
            return parent
        def createChild(child_val, val_node_dict):
            child = TreeNode(child_val)
            val_node_dict[child_val] = child
            return child
        
        val_node_dict = {}
        have_parent_nodes = set()
        all_nodes = set()
        for description in descriptions:
            parent_val, child_val, isLeft = description
            parent = createParent(parent_val, val_node_dict) if parent_val not in val_node_dict else \
                     val_node_dict[parent_val]
            child = createChild(child_val, val_node_dict) if child_val not in val_node_dict else \
                    val_node_dict[child_val]
            if isLeft:
                parent.left = child
            else:
                parent.right = child
            all_nodes.add(child_val)
            all_nodes.add(parent_val)
            have_parent_nodes.add(child_val)
        return val_node_dict[(all_nodes-have_parent_nodes).pop()]

#2096. Step-By-Step Directions From a Binary Tree Node to Another
class GetDirections(object):
    def getDirections(self, root, startValue, destValue):
        def buildRoute(startRoute, destRoute):
            while startRoute and destRoute:
                if startRoute[0] != destRoute[0]:
                    break
                startRoute = startRoute[1:]
                destRoute = destRoute[1:]
            return 'U' * len(startRoute) + destRoute
                    
        def findValInTree(root, val):
            dfs_stack = deque([(root, '')])
            while dfs_stack:
                node, route = dfs_stack.pop()
                if node.val == val:
                    return route
                if node.left:
                    dfs_stack.append((node.left, route + 'L'))
                if node.right:
                    dfs_stack.append((node.right, route + 'R'))
            return ''

        startRoute = findValInTree(root, startValue)
        destRoute = findValInTree(root, destValue)
        return buildRoute(startRoute, destRoute)

#1110. Delete Nodes And Return Forest
class DelNodes(object):
    def delNodes(self, root, to_delete):
        to_delete_set = set(to_delete)
        res_dict = {root.val: root}
        dfs_stack = deque([(root, 'ROOT', None)])
        while dfs_stack:
            node, node_type, prev_node = dfs_stack.pop()
            if node.left:
                dfs_stack.append((node.left, 'LEFT', node))
            if node.right:
                dfs_stack.append((node.right, 'RIGHT', node))
            if node.val in to_delete_set:
                if node_type == 'RIGHT':
                    prev_node.right = None
                elif node_type == 'LEFT':
                    prev_node.left = None
                if node.val in res_dict:
                    res_dict.pop(node.val)
                if node.left:
                    res_dict[node.left.val] = node.left
                if node.right:
                    res_dict[node.right.val] = node.right
        return res_dict.values()

#1530. Number of Good Leaf Nodes Pairs
class CountPairs(object):
    def countPairs(self, root, distance):
        def getLeafRoute(root):
            dfs_stack = deque([(root, '')])
            leaf_route_arr = []
            while dfs_stack:
                node, route = dfs_stack.pop()
                if node.left:
                    dfs_stack.append((node.left, route + 'L'))
                if node.right:
                    dfs_stack.append((node.right, route + 'R'))
                if not node.right and not node.left:
                    leaf_route_arr.append( route)
            return leaf_route_arr

        route_arr = getLeafRoute(root)
        result = 0
        for route_index, route in enumerate(route_arr[:-1]):
            for next_route in route_arr[route_index+1:]:
                min_route_len = min(len(route), len(next_route))
                max_route_len = max(len(route), len(next_route))
                for index in range(min_route_len):
                    if route[index] != next_route[index]:
                        dist = 2 * (min_route_len - index) + (max_route_len - min_route_len)
                        if dist <= distance:
                            result += 1
                        break
        return result

#1190. Reverse Substrings Between Each Pair of Parentheses
class ReverseParentheses(object):
    def reverseParentheses(self, s):
        def search(string, target):
            for index, char in enumerate(string):
                if char == target:
                    return index
        def checkString(string):
            for char in string:
                if char == '(':
                    return True
            return False

        def checkChars(string):
            for char in string:
                if char != '(' and char != ')':
                    return True
            return False
            
        if not checkString(s):
            return s
        first_par_index = search(s, '(')
        last_par_index = len(s) - 1 - search(s[::-1], ')')
        prefix = s[:first_par_index]
        postfix = s[last_par_index+1:]
        if not checkChars(s[first_par_index:last_par_index+1]):
            return prefix + postfix
        s = s[first_par_index:last_par_index+1]
        string_arr = list(filter(lambda x: x, re.split(r'[()]', s)))
        par_arr = list(filter(lambda x: x, re.split(r'[a-z]', s)))[:-1]
        degree = -1
        string_matrix = []
        for index in range(len(string_arr)):
            for par in par_arr[index]:
                degree += 1 if par == '(' else -1
            while len(string_matrix) <= degree:
                string_matrix.append([None for counter in range(len(string_arr))])
            string_matrix[degree][index] = string_arr[index]
        for row_index in range(len(string_matrix) - 1, 0, -1):
            merge_arr, merge_index = deque([]), 0
            for col_index in range(len(string_matrix[row_index])):
                if string_matrix[row_index][col_index]:
                    if not merge_arr:
                        merge_index = col_index
                    merge_arr.appendleft(string_matrix[row_index][col_index][::-1])
                if not string_matrix[row_index][col_index] and merge_arr:
                    for index in range(len(merge_arr)):
                        string_matrix[row_index-1][merge_index+index] = merge_arr[index]
                    merge_arr = []
            if merge_arr:
                for index in range(len(merge_arr)):
                        string_matrix[row_index-1][merge_index+index] = merge_arr[index]    
        result = ''.join(reversed(''.join(string_matrix[0])))
        return prefix + result + postfix

#1380. Lucky Numbers in a Matrix
class LuckyNumbers(object):
    def luckyNumbers(self, matrix):
        max_in_column = matrix[0]
        min_in_row = [min(matrix[0])]
        for row_index in range(1, len(matrix)):
            min_num = matrix[row_index][0]
            for col_index in range(len(matrix[row_index])):
                min_num = min(min_num, matrix[row_index][col_index])
                max_in_column[col_index] = max(max_in_column[col_index], matrix[row_index][col_index])
            min_in_row.append(min_num)
        return list(set(max_in_column) & set(min_in_row))

#1395. Count Number of Teams
class NumTeams(object):
    def numTeams(self, rating):
        soldier_graph = {}
        for cur_soldier in rating:
            for prev_soldier in soldier_graph:
                if prev_soldier < cur_soldier:
                    soldier_graph[prev_soldier]['U'].append(cur_soldier)
                else:
                    soldier_graph[prev_soldier]['L'].append(cur_soldier)
            soldier_graph[cur_soldier] = {'U': [], 'L': []}

        result = 0
        for first_soldier in soldier_graph:
            upper_soldier_arr = soldier_graph[first_soldier]['U']
            lower_soldier_arr = soldier_graph[first_soldier]['L']
            for upper_soldier in upper_soldier_arr:
                result += len(soldier_graph[upper_soldier]['U'])
            for lower_soldier in lower_soldier_arr:
                result += len(soldier_graph[lower_soldier]['L'])
        return result

#3016. Minimum Number of Pushes to Type Word II
class MinimumPushes(object):
    def minimumPushes(self, word):
        letter_freq = {}
        for char in word:
            if char not in letter_freq:
                letter_freq[char] = 1
            else:
                letter_freq[char] += 1
        freq_amount = {}
        for char in letter_freq:
            if letter_freq[char] not in freq_amount:
                freq_amount[letter_freq[char]] = 1
            else:
                freq_amount[letter_freq[char]] += 1
        result, lvl, button_amount, button_left = 0, 1, 8, 8
        for freq in sorted(freq_amount.keys(), reverse=True):
            char_amount = freq_amount[freq]
            while True:
                char_amount -= button_left
                if char_amount >= 0:
                    result += lvl * button_left * freq
                    lvl += 1
                    button_left = button_amount
                else:
                    result += lvl * (button_left + char_amount) * freq
                    button_left = abs(char_amount)
                    break 
        return result

#273. Integer to English Words
class numberToWords(object):
    def numberToWords(self, num):
        if num == 0:
            return 'Zero'
            
        numbers = {'1': 'One',
                   '2': 'Two',
                   '3': 'Three',
                   '4': 'Four',
                   '5': 'Five',
                   '6': 'Six',
                   '7': 'Seven',
                   '8': 'Eight',
                   '9': 'Nine'}
        numbers_order = {'1': {'0': 'Ten',
                               '1': 'Eleven',
                               '2': 'Twelve',
                               '3': 'Thirteen',
                               '4': 'Fourteen',
                               '5': 'Fifteen',
                               '6': 'Sixteen',
                               '7': 'Seventeen',
                               '8': 'Eighteen',
                               '9': 'Nineteen'},
                         '2': 'Twenty',
                         '3': 'Thirty',
                         '4': 'Forty',
                         '5': 'Fifty',
                         '6': 'Sixty',
                         '7': 'Seventy',
                         '8': 'Eighty',
                         '9': 'Ninety'}
        degree = deque(['', 'Thousand', 'Million', 'Billion'])
        result = deque([])
        while num:
            num_str = str(num % 1000)[::-1]
            res = deque([])
            for index, number in enumerate(num_str):
                if (index == 0) and (number in numbers):
                    res.appendleft(numbers[number])
                if (index == 1) and (number in numbers_order):
                    if number == '1':
                        if len(res) > 0:
                            res.popleft()
                        res.appendleft(numbers_order[number][num_str[index-1]])
                    else:
                        res.appendleft(numbers_order[number])
                if (index == 2) and (number in numbers):
                    res.appendleft('Hundred')
                    res.appendleft(numbers[number])
            result.append(' '.join(res))
            num //= 1000
        res_arr = deque([])
        for index in range(len(result)):
            if result[index]:
                res_string = result[index]
                if degree[index]:
                    res_string += ' ' + degree[index]
                res_arr.appendleft(res_string)
        return ' '.join(res_arr)

#885. Spiral Matrix III
class SpiralMatrixIII(object):
    def spiralMatrixIII(self, rows, cols, rStart, cStart):
        result = [[rStart, cStart]]
        lvl = 1
        while len(result) != rows * cols:
            #Right col
            for row in range(rStart - lvl + 1, rStart + lvl + 1):
                cell = [row, cStart + lvl]
                if (0 <= cell[0] < rows) and (0 <= cell[1] < cols):
                    result.append(cell)
            #Lower row
            for col in range(cStart + lvl - 1, cStart - lvl - 1, -1):
                cell = [rStart + lvl, col]
                if (0 <= cell[0] < rows) and (0 <= cell[1] < cols):
                    result.append(cell)
            #Left col
            for row in range(rStart + lvl - 1, rStart - lvl - 1, -1):
                cell = [row, cStart - lvl]
                if (0 <= cell[0] < rows) and (0 <= cell[1] < cols):
                    result.append(cell)
            #Upper row
            for col in range(cStart - lvl + 1, cStart + lvl + 1):
                cell = [rStart - lvl, col]
                if (0 <= cell[0] < rows) and (0 <= cell[1] < cols):
                    result.append(cell)
            lvl += 1
        return result

#840. Magic Squares In Grid
class NumMagicSquaresInside(object):
    def numMagicSquaresInside(self, grid):
        def getCellInfo(calc_matrix, grid, index_row, index_col):
            val = grid[index_row][index_col]
            cell_info = {}
            #Левая диагональ
            if (index_row - 1 >= 0) and (index_col - 1 >= 0):
                cell_info['left_diag'] = val + calc_matrix[index_row-1][index_col-1]['left_diag']
                if (index_row - 3 >= 0) and (index_col - 3 >= 0):
                    cell_info['left_diag'] -= grid[index_row-3][index_col-3]
            else:
                cell_info['left_diag'] = val
            #Столбец
            if (index_row - 1 >= 0):
                cell_info['col'] = val + calc_matrix[index_row-1][index_col]['col']
                if (index_row - 3 >= 0):
                    cell_info['col'] -= grid[index_row-3][index_col]
            else:
                cell_info['col'] = val
            #Строка
            if (index_col - 1 >= 0):
                cell_info['row'] = val + calc_matrix[index_row][index_col-1]['row']
                if (index_col - 3 >= 0):
                    cell_info['row'] -= grid[index_row][index_col-3]
            else:
                cell_info['row'] = val
            #Правая диагональ
            if (index_row - 1 >= 0) and (index_col + 1 < m):
                cell_info['right_diag'] = val + calc_matrix[index_row-1][index_col+1]['right_diag']
                if (index_row - 3 >= 0) and (index_col + 3 < m):
                    cell_info['right_diag'] -= grid[index_row-3][index_col+3]
            else:
                cell_info['right_diag'] = val
            return cell_info
            
        def getIncludeInfo(include_matrix, grid, index_row, index_col):
            val = grid[index_row][index_col]
            include_info = {}
            if (index_col - 1 >= 0):
                include_info = dict(include_matrix[index_row][index_col-1])
                if val in include_info:
                    include_info[val] += 1
                else:
                    include_info[val] = 1
                if (index_col - 3 >= 0):
                    include_info[grid[index_row][index_col-3]] -= 1
                    if not include_info[grid[index_row][index_col-3]]:
                        include_info.pop(grid[index_row][index_col-3])
            else:
                include_info[val] = 1
            return include_info

        def checkValidness(include_matrix, corner_row, corner_col):
            include_set = set()
            for counter in range(3):
                for num in include_matrix[corner_row-counter][corner_col].keys():
                    if (num == 0) or (num > 9) or (include_matrix[corner_row-counter][corner_col][num] > 1) or (num in include_set):
                        return False
                    include_set.add(num)
            return True

        def checkCommonNum(calc_matrix, corner_row, corner_col):
            common_num = calc_matrix[corner_row][corner_col]['left_diag']
            if common_num != calc_matrix[corner_row][corner_col-2]['right_diag']:
                return False
            for counter in range(3):
                if common_num != calc_matrix[corner_row][corner_col-counter]['col']:
                    return False
                if common_num != calc_matrix[corner_row-counter][corner_col]['row']:
                    return False
            return True
        
        n, m = len(grid), len(grid[0])
        if (n < 3) or (m < 3):
            return 0

        calc_matrix, include_matrix = [], []
        for index_row in range(n):
            calc_matrix.append([])
            include_matrix.append([])
            for index_col in range(m):
                calc_matrix[index_row].append(getCellInfo(calc_matrix, grid, index_row, index_col))
                include_matrix[index_row].append(getIncludeInfo(include_matrix, grid, index_row, index_col))

        result = 0
        for corner_row in range(2, n):
            for corner_col in range(2, m):
                if checkValidness(include_matrix, corner_row, corner_col):
                    result += checkCommonNum(calc_matrix, corner_row, corner_col) 

        return result

#703. Kth Largest Element in a Stream
class KthLargest(object):

    def __init__(self, k, nums):
        self.nums = list(sorted(nums))
        self.k = k

    def add(self, val):
        def binarySearch(nums, val):
            left_index, right_index = 0, len(nums) - 1
            if val <= nums[left_index]:
                return left_index
            if val >= nums[right_index]:
                return right_index + 1
            while right_index - left_index > 1:
                middle_index = (left_index + right_index) // 2
                if nums[middle_index] == val:
                    right_index = middle_index
                    break
                if nums[middle_index] > val:
                    right_index = middle_index
                if nums[middle_index] < val:
                    left_index = middle_index
            return right_index 
            
        if not self.nums:
            self.nums.append(val)
        else:
            self.nums.insert(binarySearch(self.nums, val), val)
        return self.nums[len(self.nums) - self.k]

#2751. Robot Collisions
class SurvivedRobotsHealths(object):
    def survivedRobotsHealths(self, positions, healths, directions):
        robots = [[positions[index], healths[index], directions[index], index] for index in range(len(healths))]
        robots.sort(key=(lambda x: x[0]))
        survive_robots, stack = [], deque([])
        for robot in robots:
            if not stack:
                if robot[2] == 'L':
                    survive_robots.append(robot)
                else:
                    stack.append(robot)
            else:
                if stack[-1][2] != robot[2]:
                    survive_flag = True
                    while stack:
                        if stack[-1][1] < robot[1]:
                            stack.pop()
                            robot[1] -= 1
                        elif stack[-1][1] > robot[1]:
                            stack[-1][1] -= 1
                            survive_flag = False
                            break
                        else:
                            stack.pop()
                            survive_flag = False
                            break
                    if survive_flag:
                        survive_robots.append(robot)
                else:
                    stack.append(robot)
        survive_robots.extend(stack)
        survive_robots.sort(key=(lambda x: x[3]))
        return [robot[1] for robot in survive_robots]

#40. Combination Sum II
class CombinationSum2(object):
    def combinationSum2(self, candidates, target):
        candidates.sort()
        combinations = set()
        combinations.add(((), 0))
        result = set()
        for new_num in candidates:
            temp_combinations = set()
            for combination in combinations:
                if combination not in temp_combinations:
                    temp_combinations.add(combination)
                
                cur_nums, cur_sum = combination
                new_nums = cur_nums + (new_num,)
                new_sum = cur_sum + new_num
                if new_sum < target:
                    new_combination = (new_nums, new_sum)
                    if new_combination not in temp_combinations:
                        temp_combinations.add(new_combination)
                if new_sum == target:
                    result.add(new_nums)
            combinations = temp_combinations
        return [list(res) for res in result]

#860. Lemonade Change
class LemonadeChange(object):
    def lemonadeChange(self, bills):
        five_bills_amount = 0
        ten_bills_amount = 0
        for bill in bills:
            if bill == 5:
                five_bills_amount += 1
            elif bill == 10:
                five_bills_amount -= 1
                ten_bills_amount += 1
                if five_bills_amount < 0:
                    return False
            else:
                if not ten_bills_amount:
                    five_bills_amount -= 3
                else:
                    five_bills_amount -= 1
                    ten_bills_amount -= 1
                if (five_bills_amount < 0) or (ten_bills_amount < 0):
                    return False
        return True

#719. Find K-th Smallest Pair Distance
class SmallestDistancePair(object):
    def smallestDistancePair(self, nums, k):
        def getLessAmount(prefix_arr, lim):
            result = 0
            left_pointer, right_pointer = 0, 0
            while right_pointer < len(prefix_arr):
                if left_pointer == right_pointer:
                    right_pointer += 1
                    continue
                if prefix_arr[right_pointer] - prefix_arr[left_pointer] > lim:
                    left_pointer += 1
                else:
                    result += right_pointer - left_pointer
                    right_pointer += 1
            return result
            
        def search(prefix_arr, pos):
            left_lim, right_lim = prefix_arr[0], prefix_arr[-1]
            while right_lim - left_lim > 1:
                middle_lim = (left_lim + right_lim) // 2
                if getLessAmount(prefix_arr, middle_lim) >= pos:
                    right_lim = middle_lim
                else:
                    left_lim = middle_lim
            left_lim_amount = getLessAmount(prefix_arr, left_lim)
            right_lim_amount = getLessAmount(prefix_arr, right_lim)
            if pos <= left_lim_amount:
                return left_lim
            else:
                return right_lim

        nums.sort()
        prefix_arr = []
        for num in nums:
            prefix_arr.append(num - nums[0])
        return search(prefix_arr, k)

#624. Maximum Distance in Arrays
class MaxDistance(object):
    def maxDistance(self, arrays):
        min_0, min_1, max_0, max_1 = None, None, None, None
        if arrays[0][0] < arrays[1][0]:
            min_0, min_1 = (arrays[0][0], 0), (arrays[1][0], 1)
        else:
            min_0, min_1 = (arrays[1][0], 1), (arrays[0][0], 0)
        if arrays[0][-1] > arrays[1][-1]:
            max_0, max_1 = (arrays[0][-1], 0), (arrays[1][-1], 1)
        else:
            max_0, max_1 = (arrays[1][-1], 1), (arrays[0][-1], 0)
            
        for index in range(2, len(arrays)):
            min_num, max_num = arrays[index][0], arrays[index][-1]
            if min_num <= min_0[0]:
                min_1 = min_0
                min_0 = (min_num, index)
            elif min_0[0] < min_num < min_1[0]:
                min_1 = (min_num, index)
            if max_num >= max_0[0]:
                max_1 = max_0
                max_0 = (max_num, index)
            elif max_1[0] < max_num < max_0[0]:
                max_1 = (max_num, index)

        if max_0[1] != min_0[1]:
            return max_0[0] - min_0[0]
        else:
            return max(max_0[0] - min_1[0], max_1[0] - min_0[0])

#959. Regions Cut By Slashes
class RegionsBySlashes(object):
    def regionsBySlashes(self, grid):
        def addBorder(border):
            if border == '\\':
                return [1, 0]
            elif border == '/':
                return [0, -1]
            else:
                return [0, 0]
                
        def getNewRow(grid_row):
            new_row = []
            for col in grid_row:
                new_row.extend(addBorder(col))
            return new_row

        def getIntermRow(main_row):
            new_row = [0 for counter in range(len(main_row))]
            for index, num in enumerate(main_row):
                if not new_row[index+num] and num:
                    new_row[index+num] = num
            return new_row

        def goRegion(regions_map, index_row, index_col):
            regions_map[index_row][index_col] = 2
            if ((index_row < len(regions_map) - 1) 
              and not regions_map[index_row+1][index_col]):
                goRegion(regions_map, index_row+1, index_col)
            if ((index_row > 0) 
              and not regions_map[index_row-1][index_col]):
                goRegion(regions_map, index_row-1, index_col)
            if ((index_col < len(regions_map) - 1) 
              and not regions_map[index_row][index_col+1]):
                goRegion(regions_map, index_row, index_col+1)
            if ((index_col > 0) 
              and not regions_map[index_row][index_col-1]):
                goRegion(regions_map, index_row, index_col-1)
            if ((index_row < len(regions_map) - 1)
              and (index_col < len(regions_map) - 1)
              and not regions_map[index_row+1][index_col+1]
              and ((regions_map[index_row][index_col+1] != -1)
                   or (regions_map[index_row+1][index_col] != -1))):
                goRegion(regions_map, index_row+1, index_col+1)
            if ((index_row < len(regions_map) - 1)
              and (index_col > 0)
              and not regions_map[index_row+1][index_col-1]
              and ((regions_map[index_row][index_col-1] != 1)
                   or (regions_map[index_row+1][index_col] != 1))):
                goRegion(regions_map, index_row+1, index_col-1)
            if ((index_row > 0)
              and (index_col > 0)
              and not regions_map[index_row-1][index_col-1]
              and ((regions_map[index_row][index_col-1] != -1)
                   or (regions_map[index_row-1][index_col] != -1))):
                goRegion(regions_map, index_row-1, index_col-1)
            if ((index_row > 0)
              and (index_col < len(regions_map) - 1)
              and not regions_map[index_row-1][index_col+1]
              and ((regions_map[index_row][index_col+1] != 1)
                   or (regions_map[index_row-1][index_col] != 1))):
                goRegion(regions_map, index_row-1, index_col+1)
                    
        regions_map = []
        for row in grid:
            regions_map.append(getNewRow(row))
            regions_map.append(getIntermRow(regions_map[-1]))
        result = 0
        for index_row in range(len(regions_map)):
            for index_col in range(len(regions_map)):
                if not regions_map[index_row][index_col]:
                    goRegion(regions_map, index_row, index_col)
                    result += 1
        return result

#650. 2 Keys Keyboard
class MinSteps(object):
    def minSteps(self, n):
        result = 0
        buffer_val = 1
        cur_amount = 1
        while cur_amount != n:
            copy_operation = 0
            if cur_amount == 1:
                copy_operation = 1  
            elif not n % cur_amount:
                copy_operation = 1
                buffer_val = cur_amount
            result += 1 + copy_operation
            cur_amount += buffer_val
        return result

#664. Strange Printer
class StrangePrinter(object):
    def _create_char_index(self, string):
        char_index = {}
        for index, char in enumerate(string):
            if char not in char_index:
                char_index[char] = [index, index]
            else:
                char_index[char][1] = index
        return char_index

    
    def strangePrinter(self, s):
        char_index = self._create_char_index(s)
        return char_index

#476. Number Complement
class FindComplement(object):
    def findComplement(self, num):
        degree = 0
        origin_num = num
        while num > 1:
            num //= 2
            degree += 1
        return origin_num ^ (2 ** (degree + 1) - 1)

#592. Fraction Addition and Subtraction
class FractionAddition(object):
    def _addFraction(self, term, fractions_arr):
        fraction = term.split('/')
        if fraction[0][0] == '+':
            fractions_arr[0].append(int(fraction[0][1:]))
        else:
            fractions_arr[0].append(int(fraction[0]))
        fractions_arr[1].append(int(fraction[1]))
        
    def _getTermArr(self, expression):
        fractions_arr = [[] , []]
        term = expression[0]
        for char in expression[1:]:
            if char in ('+', '-'):
                self._addFraction(term, fractions_arr)
                term = char
            else:
                term += char
        self._addFraction(term, fractions_arr)
        return fractions_arr

    def _findFractionsSum(self, numerators, denominators):
        res_denominator = reduce(lambda a, b: a * b, denominators)
        res_numerator = sum([numerators[index] * res_denominator // denominators[index] 
                         for index in range(len(numerators))])
        return [res_numerator, res_denominator]

    def _finGCD(self, num_1, num_2):
        greater_num = max(num_1, num_2)
        lower_num = min(num_1, num_2)
        while lower_num:
            while greater_num >= 0:
                greater_num -= lower_num
            remain = lower_num + greater_num
            greater_num = lower_num 
            lower_num = remain
        return greater_num
        
    def fractionAddition(self, expression):
        if expression[0] != '-':
            expression = '+' + expression
        fractions_arr = self._getTermArr(expression)
        res_fraction = self._findFractionsSum(fractions_arr[0], fractions_arr[1])
        gcd = self._finGCD(abs(res_fraction[0]), res_fraction[1])
        res_fraction = list(map(lambda x: x // gcd, res_fraction))
        return str(res_fraction[0]) + '/' + str(res_fraction[1])

#590. N-ary Tree Postorder Traversal
class Postorder(object):        
    def postorder(self, root):
        result = []
        dfs_stack = deque([root])
        while dfs_stack:
            node = dfs_stack.pop()
            result.append(node.val)
            if node.children:
                dfs_stack.extend(node.children)
        result.reverse()
        return result    

#1514. Path with Maximum Probability
class MaxProbability(object):
    def _getGraph(self, uEdges, uSuccProb):
        graph = defaultdict(list)
        for index, edge in enumerate(uEdges):
            graph[edge[0]].append((uSuccProb[index], edge[1]))
            graph[edge[1]].append((uSuccProb[index], edge[0]))
        return graph
        
    def maxProbability(self, n, edges, succProb, start_node, end_node):
        graph = self._getGraph(edges, succProb)
        node_prob = {index: -1 if index == start_node else 0 for index in range(n)}
        max_node = []
        heapq.heapify(max_node)
        heapq.heappush(max_node, (node_prob[start_node], start_node))
        while max_node:
            cur_prob, cur_node = heapq.heappop(max_node)
            if cur_node in graph and cur_node != end_node:
                for next_prob, next_node in graph[cur_node]:
                    if next_prob * cur_prob < node_prob[next_node]:
                        node_prob[next_node] = next_prob * cur_prob
                        heapq.heappush(max_node, (node_prob[next_node], next_node))
        return node_prob[end_node] * -1

#145. Binary Tree Postorder Traversal
class PostorderTraversal(object):
    def postorderTraversal(self, root):
        dfs_stack = deque([])
        if root:
            dfs_stack.append(root)
        result = []
        right_node = False
        while dfs_stack:
            cur_node = dfs_stack.pop() if not right_node else dfs_stack.popleft()
            result.append(cur_node.val)
            right_node = False
            if cur_node.right:
                dfs_stack.appendleft(cur_node.right)
                right_node = True
            if cur_node.left:
                dfs_stack.append(cur_node.left)
        result.reverse()
        return result

#1905. Count Sub Islands
class СountSubIslands(object):
    def _goGrid(self, grid, main_grid, start_row_index, start_col_index):
        sub_island_flag = True
        dfs_stack = deque([[start_row_index, start_col_index]])
        while dfs_stack:
            coord = dfs_stack.pop()
            row_index, col_index = coord[0], coord[1]
            grid[row_index][col_index] = 0
            if not main_grid[row_index][col_index]:
                sub_island_flag = False
            if (row_index > 0) and (grid[row_index-1][col_index]):
                dfs_stack.append([row_index-1, col_index])
            if (row_index < len(grid) - 1) and (grid[row_index+1][col_index]):
                dfs_stack.append([row_index+1, col_index])
            if (col_index > 0) and (grid[row_index][col_index-1]):
                dfs_stack.append([row_index, col_index-1])
            if (col_index < len(grid[0]) - 1) and (grid[row_index][col_index+1]):
                dfs_stack.append([row_index, col_index+1])
        return sub_island_flag
            
    def countSubIslands(self, grid1, grid2):
        result = 0
        for row_index in range(len(grid2)):
            for col_index in range(len(grid2[0])):
                if grid2[row_index][col_index]:
                    result += self._goGrid(grid2, grid1, row_index, col_index)
        return result

#947. Most Stones Removed with Same Row or Column
class RemoveStones(object):
    def _getGraph(self, uStones):
        graph = {}
        for stone in uStones:
            graph[tuple(stone)] = set()
        for index, main_stone in enumerate(uStones[:-1]):
            for add_stone in uStones[index+1:]:
                if (main_stone[0] == add_stone[0]) or (main_stone[1] == add_stone[1]):
                    stone_1, stone_2 = tuple(main_stone), tuple(add_stone)
                    graph[stone_1].add(stone_2)
                    graph[stone_2].add(stone_1)
        return graph

    def _dfs(self, uGraph):
        dfs_stack = deque([list(uGraph.keys())[0]])
        while dfs_stack:
            stone = dfs_stack.pop()
            if stone in uGraph:
                dfs_stack.extend(uGraph[stone])
                uGraph.pop(stone)
                    
    def removeStones(self, stones):
        graph = self._getGraph(stones)
        remain_stones = 0
        while graph:
            self._dfs(graph)
            remain_stones += 1
        return len(stones) - remain_stones

#1894. Find the Student that Will Replace the Chalk
class ChalkReplacer(object):
    def _getPrefixSum(self, uChalk):
        prefix_sum = [uChalk[0]]
        for num in uChalk[1:]:
            prefix_sum.append(prefix_sum[-1] + num)
        return prefix_sum

    def _binarySearch(self, uPrefixSum, uK):
        if uK < uPrefixSum[0]:
            return 0
        left_pointer, right_pointer = 0, len(uPrefixSum) -  1
        while right_pointer - left_pointer > 1:
            middle_pointer = (right_pointer + left_pointer) // 2
            if uPrefixSum[middle_pointer] <= uK:
                left_pointer = middle_pointer
            else:
                right_pointer = middle_pointer
        return right_pointer
        
    def chalkReplacer(self, chalk, k):
        prefix_sum = self._getPrefixSum(chalk)
        k -= prefix_sum[-1] * (k // prefix_sum[-1])
        return self._binarySearch(prefix_sum, k)

#2699. Modify Graph Edge Weights
class ModifiedGraphEdges(object):
    def _getGraph(self, pN, pEdges):
        graph = {index: dict() for index in range(pN)}
        for edge in pEdges:
            graph[edge[0]][edge[1]] = edge[2]
            graph[edge[1]][edge[0]] = edge[2]
        return graph

    def _getResGraph(self, pGraph, pSource, pDestination, pTarget):
        modified_edges = deque([])
        res_graph = {index: dict() for index in pGraph}
        for cur_node in pGraph:
            for next_node in pGraph[cur_node]:
                if pGraph[cur_node][next_node] == -1:
                    if cur_node < next_node:
                        modified_edges.append([cur_node, next_node])
                    res_graph[cur_node][next_node] = 1
                else:
                    res_graph[cur_node][next_node] = pGraph[cur_node][next_node]
        while True:
            min_path = self._algoDijkstra(res_graph, pSource, pDestination)
            if min_path == pTarget:
                return res_graph
            elif modified_edges:
                mod_edge = modified_edges.popleft()
                res_graph[mod_edge[0]][mod_edge[1]] += pTarget - min_path
                res_graph[mod_edge[1]][mod_edge[0]] += pTarget - min_path
            else: 
                return {}
        
    def _algoDijkstra(self, pGraph, pSource, pDestination):
        min_node_vals = []
        node_vals = {index: None for index in pGraph if index != pSource}
        checked_nodes = set()
        heapq.heapify(min_node_vals)
        heapq.heappush(min_node_vals, (0, pSource))
        while min_node_vals:
            cur_node_val, cur_node = heapq.heappop(min_node_vals)
            if cur_node != pDestination:
                checked_nodes.add(cur_node)
                for next_node in pGraph[cur_node]:
                    if (next_node not in checked_nodes
                        and pGraph[cur_node][next_node] != -1
                        and (node_vals[next_node] is None 
                             or pGraph[cur_node][next_node] + cur_node_val < node_vals[next_node])):
                        node_vals[next_node] = pGraph[cur_node][next_node] + cur_node_val
                        heapq.heappush(min_node_vals, (node_vals[next_node], next_node))
        return node_vals[pDestination]  

    def _getEdges(self, graph):
        result = []
        for cur_node in graph:
            for next_node in graph[cur_node]:
                if cur_node < next_node:
                    result.append([cur_node, next_node, graph[cur_node][next_node]])
        return result
    
    def modifiedGraphEdges(self, n, edges, source, destination, target):
        graph = self._getGraph(n, edges)
        min_path = self._algoDijkstra(graph, source, destination)
        if min_path and min_path < target:
            return []
        else:
            res_graph = self._getResGraph(graph, source, destination, target)
            return self._getEdges(res_graph)

#1945. Sum of Digits of String After Convert
class GetLucky(object):
    def getLucky(self, s, k):
        num_str = []
        for char in s:
            num_str.extend(str(ord(char) - 96))
        index = 0
        while True:
            num = reduce(lambda x, y: x + int(y), num_str, 0)
            index += 1
            if index == k:
                return num
            num_str = []
            num_str.extend(str(num))

#2028. Find Missing Observations
class MissingRolls(object):
    def missingRolls(self, rolls, mean, n):
        unknown_rolls_sum = mean * (n + len(rolls)) - sum(rolls)
        if 1.0 <= (unknown_rolls_sum / n) <= 6.0:
            result = [unknown_rolls_sum // n for _ in range(n)]
            ostat = unknown_rolls_sum - result[0] * n
            index = 0
            while ostat:
                if ostat > 6 - result[index]:
                    ostat -= 6 - result[index]
                    result[index] = 6
                else:
                    result[index] += ostat
                    ostat = 0
                index += 1
            return result
        else:
            return []

#874. Walking Robot Simulation
class RobotSim(object):
    def _getObstaclesDict(self, pObstacles):
        obstacles_dict = {}
        for obstacle in pObstacles:
            if obstacle[0] not in obstacles_dict:
                obstacles_dict[obstacle[0]] = set()
            obstacles_dict[obstacle[0]].add(obstacle[1])
        return obstacles_dict
    
    def _turnAngle(self, pAngle, pCommande):
        if pCommande == -2:
            pAngle += 90
            return pAngle % 360
        else:
            pAngle -= 90
            return pAngle if pAngle >= 0 else 270

    def _changePosition(self, pPosition, pAngle, pSteps, pObstacles):
        result = pPosition
        for _ in range(pSteps):
            new_position = list(result)
            if pAngle in (0, 180):
                new_position[0] += -1 if pAngle == 180 else 1
            else:
                new_position[1] += 1 if pAngle == 90 else -1
            if new_position[0] in pObstacles and new_position[1] in pObstacles[new_position[0]]:
                return result
            result = new_position
        return result
        
    def robotSim(self, commands, obstacles):
        obstacles_dict = self._getObstaclesDict(obstacles)
        cur_angle = 90
        cur_position = [0, 0]
        result = 0
        for command in commands:
            if command < 0:
                cur_angle = self._turnAngle(cur_angle, command)
            else:
                cur_position = self._changePosition(cur_position, cur_angle, command, obstacles_dict)
                result = max(result, cur_position[0] ** 2 + cur_position[1] ** 2 )
        return result         

#3217. Delete Nodes From Linked List Present in Array
class ModifiedList(object):
    def modifiedList(self, nums, head):
        exclude_nums = set(nums)
        new_head, cur_node = None, None
        while head:
            if head.val not in exclude_nums:
                if new_head is None:
                    new_head = head
                    cur_node = new_head
                else:
                    cur_node.next = head
                    cur_node = cur_node.next
            head = head.next
        if cur_node.next and  cur_node.next.val in exclude_nums:
            cur_node.next = None
        return new_head

#2326. Spiral Matrix IV
class SpiralMatrix(object):
    def _nextCell(self, pCurRow, pCurCol, pTurn, pM, pN):
        if pCurRow == pTurn:
            pCurCol += 1
            if pCurCol > pN - 1 - pTurn:
                pCurCol -= 1
                pCurRow += 1
        elif pCurRow == pM - 1 - pTurn:
            pCurCol -= 1
        
    def spiralMatrix(self, m, n, head):
        res_matrix = [[-1 for _ in range(n)] for _ in range(m)]
        cur_turn = 0
        while cur_turn <= (m - 1) // 2:
            cur_row = cur_turn
            for cur_col in range(cur_turn, n-cur_turn):
                if not head:
                    break
                res_matrix[cur_row][cur_col] = head.val
                head = head.next
            cur_col = n - 1 - cur_turn
            for cur_row in range(cur_turn+1, m-cur_turn):
                if not head:
                    break
                res_matrix[cur_row][cur_col] = head.val
                head = head.next
            cur_row = m - 1 - cur_turn
            for cur_col in range(n-2-cur_turn, cur_turn-1, -1):
                if not head:
                    break
                res_matrix[cur_row][cur_col] = head.val
                head = head.next
            cur_col = cur_turn
            for cur_row in range(m-2-cur_turn, cur_turn, -1):
                if not head:
                    break
                res_matrix[cur_row][cur_col] = head.val
                head = head.next
            cur_turn += 1
        return res_matrix

#2807. Insert Greatest Common Divisors in Linked List
class InsertGreatestCommonDivisors(object):
    def _findGCD(self, pNum1, pNum2):
        greater, less = max(pNum1, pNum2), min(pNum1, pNum2)
        while less:
            temp_greater = less
            less = greater - (less * (greater // less))
            greater = temp_greater
        return greater
    
    def insertGreatestCommonDivisors(self, head):
        res = head
        while head:
            if head.next:
                head.next = ListNode(self._findGCD(head.val, head.next.val), head.next)
                head = head.next
            head = head.next
        return res

#1367. Linked List in Binary Tree
class IsSubPath(object):
    def _getListVals(self, pHead):
        result = ''
        while pHead:
            result += str(pHead.val)
            pHead = pHead.next
        return result
    
    def _getRoutes(self, pRoot):
        result = []
        dfs_stack = deque([(str(pRoot.val), pRoot)])
        while dfs_stack:
            cur_route, cur_node = dfs_stack.pop()
            if not cur_node.right and not cur_node.left:
                result.append(cur_route)
                continue
            if cur_node.right:
                new_route = cur_route + str(cur_node.right.val)
                dfs_stack.append((new_route, cur_node.right))
            if cur_node.left:
                new_route = cur_route + str(cur_node.left.val)
                dfs_stack.append((new_route, cur_node.left))
        return result
        
    def isSubPath(self, head, root):
        target_path = self._getListVals(head)
        routes = self._getRoutes(root)
        for route in routes:
            if route.find(target_path) >= 0:
                return True
        return False       

#725. Split Linked List in Parts
class SplitListToParts(object):
    def _getListSize(self, pHead):
        result = 0
        while pHead:
            result += 1
            pHead = pHead.next
        return result

    def _getPartsSize(self, pHeadSize, pK):
        result = [pHeadSize // pK for _ in range(pK)]
        remainder = pHeadSize % pK
        for index in range(len(result)):
            if remainder:
                result[index] += 1
                remainder -= 1
            else: 
                break
        return result

    def _getParts(self, pPartsSize, pHead):
        result = []
        cur_part = 0
        new_part_flag = True
        while pHead:
            if new_part_flag:
                result.append(pHead)
                new_part_flag = False
            pPartsSize[cur_part] -= 1
            if not pPartsSize[cur_part]:
                cur_part += 1
                new_part_flag = True
                temp_node = pHead.next
                pHead.next = None
                pHead = temp_node
            else:
                pHead = pHead.next
        for _ in range(len(pPartsSize) - cur_part):
            result.append(None)
        return result
        
    def splitListToParts(self, head, k):
        head_size = self._getListSize(head)
        parts_size = self._getPartsSize(head_size, k)
        return self._getParts(parts_size, head)

#2220. Minimum Bit Flips to Convert Number
class MinBitFlips(object):
    def minBitFlips(self, start, goal):
        start_bin, goal_bin = bin(start)[2:], bin(goal)[2:]
        if len(start_bin) > len(goal_bin):
            goal_bin = '0' * (len(start_bin) - len(goal_bin)) + goal_bin
        elif len(start_bin) < len(goal_bin):
            start_bin = '0' * (len(goal_bin) - len(start_bin)) + start_bin
        result = 0
        for index in range(len(start_bin)):
            if start_bin[index] != goal_bin[index]:
                result += 1
        return result      

#1684. Count the Number of Consistent Strings
class CountConsistentStrings(object):
    def countConsistentStrings(self, allowed, words):
        allowed_set = set(allowed)
        result = 0
        for word in words:
            word_set = set(word)
            if not word_set - allowed_set:
                result += 1
        return result                  

#1310. XOR Queries of a Subarray
class XorQueries(object):
    def _getPrefixXOR(self, pArr):
        result = [0]
        for num in pArr:
            result.append(result[-1] ^ num)
        return result
        
    def xorQueries(self, arr, queries):
        prefixXOR = self._getPrefixXOR(arr)
        result = []
        for query in queries:
            result.append(prefixXOR[query[1]+1] ^ prefixXOR[query[0]])
        return result

#539. Minimum Time Difference
class FindMinDifference(object):
    def _convertTime(self, pTime):
        time_arr = [int(time) for time in pTime.split(':')]
        return time_arr[0] * 60 + time_arr[1]
        
    def findMinDifference(self, timePoints):
        minutes_arr = [self._convertTime(time) for time in timePoints]
        minutes_arr.sort()
        result = 1440 - minutes_arr[-1] + minutes_arr[0]
        for index in range(1, len(minutes_arr)):
            result = min(result, minutes_arr[index] - minutes_arr[index-1])
        return result

#884. Uncommon Words from Two Sentences
class UncommonFromSentences(object):
    def uncommonFromSentences(self, s1, s2):
        string_set_1 = set()
        string_set_2 = set()
        for string in s1.split():
            if string in string_set_1:
                string_set_2.add(string)
            else:
                string_set_1.add(string)
        for string in s2.split():
            if string in string_set_2:
                string_set_1.add(string)
            else:
                string_set_2.add(string)
        return list(string_set_1 ^ string_set_2)

#179. Largest Number
class LargestNumber(object):
    def _getGreater(self, pLeft, pRight):
        left_right = str(pLeft[0]) + str(pRight[0])
        right_left = str(pRight[0]) + str(pLeft[0])
        for index in range(len(left_right)):
            if left_right[index] > right_left[index]:
                return pLeft.popleft()
            elif right_left[index] > left_right[index]:
                return pRight.popleft()
        return pLeft.popleft()
    
    def _merge(self, pArray, pLeftIndex, pRightIndex):
        middle_index = pLeftIndex + (pRightIndex - pLeftIndex) // 2
        left = deque(pArray[pLeftIndex:middle_index+1])
        right = deque(pArray[middle_index+1:pRightIndex+1])
        result = []
        while left and right:
            result.append(self._getGreater(left, right))
        if left:
            result.extend(left)
        else:
            result.extend(right)
        pArray[pLeftIndex:pRightIndex+1] = result
        
    def _mergeSort(self, pArray, pLeftIndex, pRightIndex):
        if pLeftIndex != pRightIndex:
            middle_index = pLeftIndex + (pRightIndex - pLeftIndex) // 2
            self._mergeSort(pArray, pLeftIndex, middle_index)
            self._mergeSort(pArray, middle_index + 1, pRightIndex)
        self._merge(pArray, pLeftIndex, pRightIndex)
        
    def largestNumber(self, nums):
        self._mergeSort(nums, 0, len(nums) - 1)
        result = ''.join(map(lambda x: str(x), nums))
        while len(result) > 1:
            if result[0] == '0':
                result = result[1:]
            else:
                break
        return result

#214. Shortest Palindrome
class ShortestPalindrome(object):
    def shortestPalindrome(self, s):
        end_index = len(s)
        while s[:end_index] != s[end_index-1::-1]:
            end_index -= 1
        return s[:end_index-1:-1] + s

#2419. Longest Subarray With Maximum Bitwise AND
class LongestSubarray(object):
    def longestSubarray(self, nums):
        max_num = max(nums)
        result = 0
        cur_length = 0
        for num in nums:
            if num == max_num:
                cur_length += 1
            else:
                result = max(result, cur_length)
                cur_length = 0
        result = max(result, cur_length)
        return result

#Сортировка слиянием
class MergeSort:
    def __init__(self, pArray):
        self._array = pArray

    def __repr__(self):
        return str(self._array)
        
    def _getGreater(self, pLeft, pRight):
        if pLeft[0] >= pRight[0]:
            return pLeft.popleft()
        else:
            return pRight.popleft()
        
    def _merge(self, pArray, pLeftIndex, pRightIndex):
        middle_index = pLeftIndex + (pRightIndex - pLeftIndex) // 2
        left = deque(pArray[pLeftIndex:middle_index+1])
        right = deque(pArray[middle_index+1:pRightIndex+1])
        result = []
        while left and right:
            result.append(self._getGreater(left, right))
        if left:
            result.extend(left)
        else:
            result.extend(right)
        pArray[pLeftIndex:pRightIndex+1] = result
        
    def _mergeSort(self, pArray, pLeftIndex, pRightIndex):
        if pLeftIndex != pRightIndex:
            middle_index = pLeftIndex + (pRightIndex - pLeftIndex) // 2
            self._mergeSort(pArray, pLeftIndex, middle_index)
            self._mergeSort(pArray, middle_index + 1, pRightIndex)
        self._merge(pArray, pLeftIndex, pRightIndex)

    def makeSort(self):
        self._mergeSort(self._array, 0, len(self._array) - 1)

#2707. Extra Characters in a String
class MinExtraChar(object):
    def _getFirstLetterDict(self, pDictionary):
        result = {}
        for word in pDictionary:
            if word[0] not in result:
                result[word[0]] = []
            result[word[0]].append(word)
        return result
        
    def minExtraChar(self, s, dictionary):
        first_letters = self._getFirstLetterDict(dictionary)
        extra_chars = {key: set() for key in range(len(s) + 1)}
        extra_chars[0] = {0}
        for index, letter in enumerate(s):
            if letter in first_letters:
                for word in first_letters[letter]:
                    if s[index:index+len(word)] == word:
                        extra_chars[index+len(word)].update(extra_chars[index])
            extra_chars[index+1].update(num + 1 for num in extra_chars[index])
        return min(extra_chars[len(s)])

#3043. Find the Length of the Longest Common Prefix
class LongestCommonPrefix(object):
    def _getLevelDict(self, pArr1):
        result = {}
        for num in pArr1:
            str_num = str(num)
            for end_index in range(len(str_num)):
                if end_index not in result:
                    result[end_index] = set()
                result[end_index].add(str_num[:end_index+1])
        return result

    def _getCommonPrefix(self, pNum, pPrefix):
        str_num = str(pNum)
        for end_index in range(len(str_num)):
            if ((end_index not in pPrefix)
                or (str_num[:end_index+1] not in pPrefix[end_index])):
                return end_index
        return len(str_num)
        
    def longestCommonPrefix(self, arr1, arr2):
        level_prefix = self._getLevelDict(arr1)
        result = 0
        for num in arr2:
            result = max(result, self._getCommonPrefix(num, level_prefix))
        return result

#2416. Sum of Prefix Scores of Strings
class SumPrefixScores(object):
    def _getPrefixScores(self, pWords):
        result = {}
        for word in pWords:
            for end_index in range(len(word)):
                prefix = word[:end_index+1]
                result.setdefault(prefix, 0)
                result[prefix] += 1
        return result
        
    def sumPrefixScores(self, words):
        prefix_scores = self._getPrefixScores(words)
        result = []
        for word in words:
            result.append(0)
            for end_index in range(len(word)):
                prefix = word[:end_index+1]
                result[-1] += prefix_scores[prefix]
        return result

#386. Lexicographical Numbers
class LexicalOrder(object):
    def _getNextNum(self, pCurNum, pMaxNum):
        next_num = pCurNum * 10
        if pCurNum == pMaxNum:
            next_num = (pCurNum // 10) + 1
        elif next_num > pMaxNum:
            next_num = (next_num // 10) + 1
        result = deque([next_num])
        while not next_num % 10:
            next_num //= 10
            result.appendleft(next_num)
        return result
        
    def lexicalOrder(self, n):
        result = [10 ** degree for degree in range(len(str(n)))]
        while len(result) < n:
            result.extend(self._getNextNum(result[-1], n))
        return result      

#729. My Calendar I
class MyCalendar:
    def __init__(self):
        self._schedule = list()

    def _dateCheck(self, pStart, pEnd):
        if (not self._schedule
            or (pStart >= self._schedule[-1][1])
            or (pEnd <= self._schedule[0][0])):
            return True
        left_index, right_index = 0, len(self._schedule) - 1
        while right_index - left_index > 1:
            middle_index = (left_index + right_index) // 2
            if pStart < self._schedule[middle_index][0]:
                right_index = middle_index
            else:
                left_index = middle_index
        if ((pStart >= self._schedule[left_index][1])
            and (pEnd <= self._schedule[right_index][0])):
            return True
        else:
            return False
        
    def book(self, start, end):
        if self._dateCheck(start, end):
            insort(self._schedule, (start, end), key=(lambda x: x[0]))
            return True
        return False       

#1381. Design a Stack With Increment Operation
class StackElem:
    def __init__(self, val, prev=None, index=0):
        self.val = val
        self.prev = prev
        self.index = index

    def __add__(self, addVal):
        self.val += addVal
        return self

class CustomStack:
    def __init__(self, maxSize):
        self.head = None
        self.maxSize = maxSize

    def push(self, x):
        if not self.head:
            self.head = StackElem(x)
        elif self.head.index < self.maxSize - 1:
            new_elem = StackElem(x, self.head, self.head.index+1)
            self.head = new_elem            

    def pop(self):
        if not self.head:
            return -1
        else:
            result = self.head.val
            self.head = self.head.prev
            return result

    def increment(self, k, val):
        cur_elem = self.head
        while cur_elem:
            if cur_elem.index < k:
                cur_elem += val
            cur_elem = cur_elem.prev

#641. Design Circular Deque
class DequeElem:
    def __init__(self, val, prev=None, next=None, index=0):
        self.val = val
        self.prev = prev
        self.next = next
        self.index = index
    
class MyCircularDeque(object):
    def __init__(self, k):
        self.head = None
        self.tail = None
        self.maxSize = k
    
    def insertFront(self, value):
        if self.isFull():
            return False
        if self.head:
            self.head.prev = DequeElem(value, None, self.head, self.head.index-1)
            self.head = self.head.prev
        else:
            self.head = DequeElem(value)
            self.tail = self.head
        return True

    def insertLast(self, value):
        if self.isFull():
            return False
        if self.tail:
            self.tail.next = DequeElem(value, self.tail, None, self.tail.index+1)
            self.tail = self.tail.next
        else:
            self.tail = DequeElem(value)
            self.head = self.tail
        return True

    def deleteFront(self):
        if self.isEmpty():
            return False
        if self.head.index == self.tail.index:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
        return True
    
    def deleteLast(self):
        if self.isEmpty():
            return False
        if self.head.index == self.tail.index:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
        return True

    def getFront(self):
        if self.isEmpty():
            return -1
        return self.head.val

    def getRear(self):
        if self.isEmpty():
            return -1
        return self.tail.val

    def isEmpty(self):
        return not self.head

    def isFull(self):
        if self.isEmpty():
            return False
        return self.tail.index - self.head.index == self.maxSize - 1

#1497. Check If Array Pairs Are Divisible by k
class CanArrange(object):
    def _getRemainderCount(self, num_arr, div):
        result = {}
        for num in num_arr:
            remainder = num % div
            result.setdefault(remainder, 0)
            result[remainder] += 1
        return result
    
    def canArrange(self, arr, k):
        if k == 1:
            return True
        remainderCount = self._getRemainderCount(arr, k)
        for remainder in remainderCount:
            if (remainder == 0) and (remainderCount[remainder] % 2 == 1):
                return False
            if ((remainder != 0)
                and ((k - remainder not in remainderCount)
                     or (remainderCount[remainder] != remainderCount[k - remainder]))):
                return False
        return True

#6. Zigzag Conversion
class Convert(object):
    def convert(self, s, numRows):
        matrix = deque([])
        cur_start = 0
        column_flag = True
        while cur_start < len(s):
            if column_flag:
                column = deque(s[cur_start:cur_start+numRows])
                matrix.append(column)
                cur_start += numRows
                column_flag = False
            else:
                if numRows == 1:
                    matrix.append([''])
                elif numRows == 2:
                    matrix.append(['', ''])
                else:
                    diag = deque([''])
                    diag.extend(s[cur_start+numRows-3:cur_start-1:-1])
                    diag.append('')
                    matrix.append(diag)
                    cur_start += numRows - 2
                column_flag = True
        if column_flag:
            matrix[-1].extendleft('' for _ in range(numRows - len(matrix[-1])))
        else:
            matrix[-1].extend('' for _ in range(numRows - len(matrix[-1])))
        result = ''
        for index_col in range(len(matrix[0])):
            for index_row in range(len(matrix)):
                if matrix[index_row][index_col]:
                    result += matrix[index_row][index_col]
        return result

#1331. Rank Transform of an Array
class ArrayRankTransform(object):
    def arrayRankTransform(self, arr):
        origin_arr = list(arr)
        arr.sort()
        num_rank = {}
        for num in arr:
            num_rank.setdefault(num, len(num_rank) + 1)
        return [num_rank[num] for num in origin_arr]        

#2491. Divide Players Into Teams of Equal Skill
class DividePlayers(object):
    def dividePlayers(self, skills):
        skill_sum = 0
        skill_amount = {}
        for skill in skills:
            skill_amount.setdefault(skill, 0)
            skill_amount[skill] += 1
            skill_sum += skill
        if skill_sum % (len(skills) // 2):
            return -1
        team_skill = skill_sum // (len(skills) // 2)
        result = 0
        checked = set()
        for skill in skill_amount:
            if skill in checked:
                continue
            if ((team_skill-skill not in skill_amount)
                or (skill_amount[skill] != skill_amount[team_skill-skill])):
                return -1
            if skill == team_skill-skill:
                if skill_amount[skill] % 2 == 1:
                    return -1
                result += (skill_amount[skill] // 2) * skill * (team_skill-skill)
            else:    
                result += skill_amount[skill] * skill * (team_skill-skill)
            checked.update({skill, team_skill-skill})
        return result

#1590. Make Sum Divisible by P
class MinSubarra(object):
    def _getTarget(self, diff, module):
        return diff if diff >= 0 else module + diff            
    
    def minSubarray(self, nums, p):
        remainder_ = sum(nums) % p
        if not remainder_:
            return 0
        prefix_remainder_, remainder_index_ = [0], {0: -1}
        result = len(nums)
        for index, num in enumerate(nums):
            prefix_remainder_.append((prefix_remainder_[-1] + num) % p)
            target = self._getTarget(prefix_remainder_[-1] - remainder_, p)
            if target in remainder_index_:
                result = min(result, index - remainder_index_[target])
            remainder_index_[prefix_remainder_[-1]] = index
        return -1 if result == len(nums) else result

#962. Maximum Width Ramp
class MaxWidthRamp:
    def maxWidthRamp(self, nums):
        num_index = {}
        for index, num in enumerate(nums):
            if num not in num_index:
                num_index[num] = [index, index]
            else:
                num_index[num][1] = index
        max_length, farthest_index = 0, 0
        for num in sorted(num_index.keys(), reverse=True):
            indexes = num_index[num]
            max_length = max(indexes[1] - indexes[0], max_length)
            max_length = max(farthest_index - indexes[0], max_length)
            farthest_index = max(indexes[1], farthest_index)
        return max_length

#1942. The Number of the Smallest Unoccupied Chair
class SmallestChair:
    def smallestChair(self, times, targetFriend):
        times_friend = [[times[index][0], times[index][1], index] 
                        for index in range(len(times))]
        times_friend.sort(key=(lambda x: x[0]))
        free_chairs = []
        occupied_chairs = 0
        leave_time = []
        for start, end, friend in times_friend:
            cur_friend_chair = None
            if leave_time:
                while leave_time and (start >= leave_time[0][0]):
                    chair = heapq.heappop(leave_time)
                    heapq.heappush(free_chairs, chair[1])
            if not free_chairs:
                cur_friend_chair = occupied_chairs
                occupied_chairs += 1
            else:
                cur_friend_chair = heapq.heappop(free_chairs)
            if friend == targetFriend:
                return cur_friend_chair
            heapq.heappush(leave_time, (end, cur_friend_chair))

#2530. Maximal Score After Applying K Operations
class MaxKelements:
    def maxKelements(self, nums, k):
        heap = []
        for num in nums:
            heapq.heappush(heap, -num)
        result = 0
        for _ in range(k):
            greatest_num = heapq.heappop(heap)
            result -= greatest_num
            heapq.heappush(heap, -((-greatest_num + 2) // 3))
        return result

#632. Smallest Range Covering Elements from K Lists
class SmallestRange:
    def smallestRange(self, nums):
        min_heap, max_num = [], None
        result = None
        for index, arr in enumerate(nums):
            heapq.heappush(min_heap, (arr[0], index))
            if max_num is None:
                max_num = arr[0]
            else:
                max_num = max(max_num, arr[0])
        indexes = {index: 0 for index in range(len(nums))}
        while True:
            min_num, arr_num = heapq.heappop(min_heap)
            if ((result is None)
                or (max_num - min_num < result[1] - result[0])):
                result = [min_num, max_num]
            indexes[arr_num] += 1
            if indexes[arr_num] == len(nums[arr_num]):
                return result
            heapq.heappush(min_heap, (nums[arr_num][indexes[arr_num]], arr_num))
            max_num = max(max_num, nums[arr_num][indexes[arr_num]])                

#2938. Separate Black and White Balls
class MinimumSteps:
    def minimumSteps(self, s):
        first_free_index = 0
        result = 0
        for index, num in enumerate(s):
            if num == '0':
                result += index - first_free_index
                first_free_index += 1
        return result

#1405. Longest Happy String
class LongestDiverseString:
    def longestDiverseString(self, a, b, c):
        letters = [(a, 'a'), (b, 'b'), (c, 'c')]
        letters.sort(reverse=True)
        substrings = []
        amount, letter = letters[0]
        while amount > 0:
            if amount > 1:
                substrings.append(letter * 2)
            else:
                substrings.append(letter)
            amount -= 2
        substring_index = 0
        for amount, letter in letters[1:]:
            while amount > 0:
                if amount > len(substrings) - substring_index:
                    substrings[substring_index] += letter * 2
                    amount -= 2
                else:
                    substrings[substring_index] += letter
                    amount -= 1
                substring_index += 1
                substring_index %= len(substrings)
        result = ''.join(substrings)
        for index in range(2, len(result)):
            if result[index] == result[index-1] == result[index-2]:
                result = result[:index]
                break
        return result

#670. Maximum Swap
class MaximumSwap:
    def maximumSwap(self, num):
        num_index = {}
        arr_num = list(str(num))
        for index, string_num in enumerate(arr_num):
            num_index.setdefault(int(string_num), list()).append(index)
        cur_index = 0
        for key in sorted(num_index.keys(), reverse=True):
            for index in num_index[key]:
                if cur_index != index:
                    arr_num[cur_index], arr_num[num_index[key][-1]] = \
                    arr_num[num_index[key][-1]], arr_num[cur_index]
                    return int(''.join(arr_num))
                cur_index += 1
        return int(''.join(arr_num))

#17. Letter Combinations of a Phone Number
class LetterCombinations:
    def letterCombinations(self, digits):
        phone = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                 '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        result = ['']
        for digit in digits:
            temp_result = []
            for string in result:
                for add_string in phone[digit]:
                    temp_result.append(string + add_string)
            result = temp_result
        return [] if len(result) == 1 else result

#2044. Count Number of Maximum Bitwise-OR Subsets
class CountMaxOrSubsets:
    def countMaxOrSubsets(self, nums):
        max_subset, result = 0, 0
        subsets = [0]
        for num in nums:
            temp_subsets = []
            for subset_value in subsets:
                include_value = subset_value | num
                exclude_value = subset_value
                if include_value > max_subset:
                    max_subset = include_value
                    result = 0
                result += include_value == max_subset
                temp_subsets.append(include_value)
                temp_subsets.append(exclude_value)
            subsets = temp_subsets
        return result

#1593. Split a String Into the Max Number of Unique Substrings
class MaxUniqueSplit(object):
    def maxUniqueSplit(self, s):
        max_subset = 0
        subsets = [(set([s[:index+1]]), index + 1) for index in range(len(s))]
        while subsets:
            temp_subsets = []
            for subset, cur_index in subsets:
                for index in range(cur_index+1, len(s)+1):
                    new_set = set(subset)
                    new_substring = s[cur_index:index]
                    if new_substring not in new_set:
                        new_set.add(new_substring)
                        temp_subsets.append((new_set, index))
            subsets = temp_subsets
            max_subset += 1
        return max_subset

#2583. Kth Largest Sum in a Binary Tree
class KthLargestLevelSum:
    def _goTree(self, node, lvl, result):
        if lvl == len(result):
            result.append(node.val)
        else:
            result[lvl] += node.val
        if node.left:
            self._goTree(node.left, lvl+1, result)
        if node.right:
            self._goTree(node.right, lvl+1, result)
            
    def kthLargestLevelSum(self, root, k):
        result = []
        self._goTree(root, 0, result)
        result.sort(reverse=True)
        return result[k-1] if k-1 < len(result) else -1

#1545. Find Kth Bit in Nth Binary String
class FindKthBit:
    def _invert(self, string):
        result = ''
        for char in string:
            if char == '0':
                result += '1'
            else:
                result += '0'
        return result
        
    def findKthBit(self, n, k):
        string = '0'
        while k > len(string):
            string += '1' + self._invert(string)[::-1]
        return string[k-1]

#2641. Cousins in Binary Tree II
class ReplaceValueInTree:
    def _goTree(self, node, lvl, penalty, scores):
        if lvl == len(scores):
            scores.append(node.val)
        else:
            scores[lvl] += node.val
        penalty_val = node.left.val if node.left else 0
        penalty_val += node.right.val if node.right else 0
        if node.left:
            penalty[node.left] = penalty_val
            self._goTree(node.left, lvl+1, penalty, scores)
        if node.right:
            penalty[node.right] = penalty_val
            self._goTree(node.right, lvl+1, penalty, scores)

    def _rebuildTree(self, node, lvl, penalty, scores):
        if node:
            node.val = scores[lvl] - penalty[node]
            self._rebuildTree(node.left, lvl+1, penalty, scores)
            self._rebuildTree(node.right, lvl+1, penalty, scores)
    
    def replaceValueInTree(self, root):
        penalty = {root: root.val}
        scores = []
        self._goTree(root, 0, penalty, scores)
        self._rebuildTree(root, 0, penalty, scores)
        return root

#951. Flip Equivalent Binary Trees
class FlipEquiv:
    def _rebuildTree(self, node1, node2, res):
        if node1.val != node2.val:
            res[0] = False
            return res
        if ((node1.left is None and node2.left is not None)
            or (node1.left is not None and node2.left is None)
            or ((node1.left and node2.left) and (node1.left.val != node2.left.val))):
            node2.left, node2.right = node2.right, node2. left
        
        if node1.left and node2.left:
            self._rebuildTree(node1.left, node2.left, res)
        elif not (node1.left is None and node2.left is None):
            res[0] = False
            return res

        if node1.right and node2.right:
            self._rebuildTree(node1.right, node2.right, res)
        elif not (node1.right is None and node2.right is None):
            res[0] = False
            return res
    
    def flipEquiv(self, root1, root2):
        if root1 and root2:
            res = [True]
            self._rebuildTree(root1, root2, res)
            return res[0]
        elif not root1 and not root2:
            return True
        else:
            return False

#1233. Remove Sub-Folders from the Filesystem
class RemoveSubfolders:
    def removeSubfolders(self, folder):
        folder_arr = []
        for sub_folder in folder:
            insort(folder_arr, sub_folder)
        
        main_folder = folder_arr[0]
        result = [main_folder]
        for sub_folder in folder_arr[1:]:
            candidate = sub_folder[:len(main_folder)]
            for char in sub_folder[len(main_folder):]:
                if char == '/':
                    break
                candidate += char
                    
            if  candidate != main_folder:
                main_folder = sub_folder
                result.append(main_folder)
        return result

#23. Merge k Sorted Lists
class MergeKLists:
    def _mergeLists(self, head1, head2):
        root = ListNode()
        cur_node = root
        root_flag = True
        while head1 and head2:
            if not root_flag:
                cur_node = cur_node.next
            else:
                root_flag = False
            if head1.val <= head2.val:
                cur_node.val = head1.val
                head1 = head1.next
            else:
                cur_node.val = head2.val
                head2 = head2.next
            cur_node.next = ListNode()
        while head1:
            cur_node = cur_node.next
            cur_node.val = head1.val
            head1 = head1.next
            cur_node.next = ListNode()
        while head2:
            cur_node = cur_node.next
            cur_node.val = head2.val
            head2 = head2.next
            cur_node.next = ListNode()
        cur_node.next = None   
        return root
                
    def mergeKLists(self, lists):
        while len(lists) > 1:
            temp_lists = []
            for index in range(0, len(lists), 2):
                head1 = lists[index]
                head2 = lists[index+1] if index < len(lists) - 1 else None
                if head1 and head2:
                    temp_lists.append(self._mergeLists(head1, head2))
                elif head1:
                    temp_lists.append(head1)
                elif head2:
                    temp_lists.append(head2)
            lists = temp_lists
        return lists[0] if lists else None

#1957. Delete Characters to Make Fancy String
class MakeFancyString:
    def makeFancyString(self, s):
        result = s[0]
        amount, first = 1, result
        for char in s[1:]:
            if first == char:
                amount += 1
            else:
                first = char
                amount = 1
            if amount <= 2:
                result += char
        return result

#3163. String Compression III
class CompressedString:
    def compressedString(self, word):
        comp = []
        cur_length, cur_char = 1, word[0]
        for char in word[1:]:
            if char == cur_char:
                if cur_length == 9:
                    comp.append(str(cur_length))
                    comp.append(cur_char)
                    cur_length = 0
                cur_length += 1
            else:
                comp.append(str(cur_length))
                comp.append(cur_char)
                cur_length = 1
                cur_char = char
        comp.append(str(cur_length))
        comp.append(cur_char)
        return ''.join(comp)

#15. 3Sum
class ThreeSum:
    def threeSum(self, nums):
        num_index, checked = {}, set()
        for num in nums:
            if num not in num_index:
                num_index[num] = 1
            else:
                num_index[num] += 1
        result = []
        for index1, num1 in enumerate(nums[:-2], 1):
            for num2 in nums[index1:-1]:
                num3 = -(num1 + num2)
                if num3 in num_index:
                    pair = tuple(sorted((num1, num2, num3)))
                    if ((((num3 != num1) and (num3 != num2))
                        or(((num1 != num2) and ((num3 == num1) or (num3 == num2))) and (num_index[num3] >= 2))
                        or ((num1 == num2 == num3) and (num_index[num3] >= 3)))
                        and pair not in checked):
                        result.append([num1, num2, num3])
                        checked.add(pair)
        return result

#2914. Minimum Number of Changes to Make Binary String Beautiful
class MinChanges:
    def minChanges(self, s):
        result = 0
        index = 0
        while index < len(s):
            zeros_amount, ones_amount = 0, 0
            for _ in range(2):
                zeros_amount += (s[index] == '0')
                ones_amount += (s[index] == '1')
                index += 1
            result += min(zeros_amount, ones_amount)
        return result

#16. 3Sum Closest
class ThreeSumClosest:
    def threeSumClosest(self, nums, target):
        nums.sort()
        result = None
        for index1 in range(len(nums) - 2):
            index2, index3 = index1 + 1, len(nums) - 1
            temp_result = None
            while index2 < index3:
                nums_sum = nums[index1] + nums[index2] + nums[index3]
                remain = target - nums_sum
                if remain > 0:
                    index2 += 1
                elif remain < 0:
                    index3 -= 1
                else:
                    return nums_sum
                if temp_result is None or temp_result[0] > abs(remain):
                    temp_result = (abs(remain), nums_sum)
            if result is None or result[0] > temp_result[0]:
                result = temp_result
        return result[1]

#3011. Find if Array Can Be Sorted
class CanSortArray:
    def _findSet(self, num):
        result = 0
        while num:
            result += num % 2
            num //= 2
        return result        
        
    def canSortArray(self, nums):
        sorted_arr = []
        border = 0
        for num in nums:
            insort_index = bisect(sorted_arr, num)
            if ((insort_index < border)
                or ((insort_index < len(sorted_arr)) 
                    and (self._findSet(num) != self._findSet(sorted_arr[insort_index])))):
                return False
            insort(sorted_arr, num)
            if ((insort_index != border) 
                and (self._findSet(num) != self._findSet(sorted_arr[insort_index-1]))):
                border = insort_index
        return True

#2070. Most Beautiful Item for Each Query
class MaximumBeauty:
    def _searchPrice(self, prices, target):
        left_index, right_index = 0, len(prices) - 1
        if target < prices[0]:
            return 0
        elif target > prices[-1]:
            return prices[-1]
        else:
            while right_index - left_index > 1:
                if prices[left_index] == target:
                    return prices[left_index]
                if prices[right_index] == target:
                    return prices[right_index]
                middle_index = (right_index + left_index) // 2
                if prices[middle_index] <= target:
                    left_index = middle_index
                else:
                    right_index = middle_index
            return prices[left_index]
            
    def maximumBeauty(self, items, queries):
        prices = []
        prices_beauty = {0: 0}
        for price, beauty in items:
            if price not in prices_beauty:
                prices_beauty[price] = beauty
                insort(prices, price)
            else:
                prices_beauty[price] = max(prices_beauty[price], beauty)
        for index in range(1, len(prices)):
            prices_beauty[prices[index]] = max(prices_beauty[prices[index]], 
                                               prices_beauty[prices[index-1]])
        result = []
        for price in queries:
            result.append(prices_beauty[self._searchPrice(prices, price)])
        return result

#2601. Prime Subtraction Operation
class PrimeSubOperation:
    def _findNearestPrime(self, primes, target):
        left_index, right_index = 0, len(primes) - 1
        if not primes or target < primes[0]:
            return None
        if target > primes[-1]:
            return right_index
        while right_index - left_index > 1:
            if primes[left_index] == target:
                return left_index
            if primes[right_index] == target:
                return right_index
            middle_index = (left_index + right_index) // 2
            if primes[middle_index] <= target:
                left_index = middle_index
            else:
                right_index = middle_index
        return left_index
        
    def _buildFirstNPrimes(self, n):
        result = list(True if (num == 2) or ((num > 2) and (num % 2 == 1)) else False
                      for num in range(n + 1))
        for index in range(3, len(result), 2):
            if result[index]:
                temp_index = index
                while index * temp_index < len(result):
                    result[index*temp_index] = False
                    temp_index += 2
        return list(index for index, res in enumerate(result) if res)
        
    def primeSubOperation(self, nums):
        primes = self._buildFirstNPrimes(max(nums) - 1)
        for index in range(len(nums)):
            nearest_prime_index = self._findNearestPrime(primes, nums[index] - 1)
            if nearest_prime_index is not None:
                if index == 0:
                    nums[index] -= primes[nearest_prime_index]
                else:
                    while ((nearest_prime_index >= 0) 
                           and (nums[index] - primes[nearest_prime_index] <= nums[index-1])):
                        nearest_prime_index -= 1
                    if nearest_prime_index >= 0:
                        nums[index] -= primes[nearest_prime_index]
            if (index != 0) and (nums[index] <= nums[index-1]):
                return False
        return True

#3105. Longest Strictly Increasing or Strictly Decreasing Subarray
class LongestMonotonicSubarray:
    def longestMonotonicSubarray(self, nums):
        result = 0
        inc_length, dec_length = 1, 1
        for index, num in enumerate(nums[1:], 1):
            if num > nums[index-1]:
                result = max(result, dec_length)
                inc_length += 1
                dec_length = 1
            elif num < nums[index-1]:
                result = max(result, inc_length)
                dec_length += 1
                inc_length = 1
            else:
                result = max(result, inc_length, dec_length)
                dec_length = 1
                inc_length = 1
        return max(result, inc_length, dec_length)

#2364. Count Number of Bad Pairs
class CountBadPairs:
    def countBadPairs(self, nums):
        diff_dict = {}
        for index, num in enumerate(nums):
            diff = num - index
            diff_dict[diff] = diff_dict.setdefault(diff, 0) + 1
        all_pairs = (len(nums) - 1) * len(nums) // 2
        good_pairs = 0
        for pairs in diff_dict.values():
            good_pairs += (pairs - 1) * pairs // 2
        return all_pairs - good_pairs

#2349. Design a Number Container System
class NumberContainers:
    def __init__(self):
        self.index_num = {}
        self.num_index = {}

    def _remove(self, index, number):
        self.num_index[number].pop(bisect.bisect_left(self.num_index[number], index))
        if not self.num_index[number]:
            self.num_index.pop(number)
    
    def change(self, index, number):
        if index in self.index_num:
            self._remove(index, self.index_num[index])
        self.index_num[index] = number
        bisect.insort(self.num_index.setdefault(number, list()), index)
        

    def find(self, number):
        return self.num_index.get(number, [-1])[0]

#3174. Clear Digits
class ClearDigits:
    def clearDigits(self, s):
        result = deque([])
        for char in s:
            if char.isdigit():
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

#1910. Remove All Occurrences of a Substring
class RemoveOccurrences:
    def removeOccurrences(self, s, part):
        while True:
            index = s.find(part)
            if index != -1:
                s = s[:index] + s[index+len(part):]
            else:
                break
        return s                

#2342. Max Sum of a Pair With Equal Sum of Digits
class MaximumSum:
    def _getDigitsSum(self, number):
        result = 0
        while number > 0:
            result += number % 10
            number //= 10
        return result
    
    def maximumSum(self, nums):
        nums.sort(reverse=True)
        digits_sum_dict = {}
        result = -1
        for num in nums:
            digits_sum = self._getDigitsSum(num)
            if digits_sum not in digits_sum_dict:
                digits_sum_dict[digits_sum] = [num, True]
            elif digits_sum_dict[digits_sum][1]:
                digits_sum_dict[digits_sum][0] += num
                digits_sum_dict[digits_sum][1] = False
                result = max(result, digits_sum_dict[digits_sum][0])
        return result

#2698. Find the Punishment Number of an Integer
class PunishmentNumber:
    def _checkNum(self, number):
        sqr_string = str(number**2)
        queue = deque([(0, int(sqr_string[0]))])
        for string in sqr_string[1:]:
            num = int(string)
            for _ in range(len(queue)):
                cur_sum, cur_num = queue.popleft()
                if cur_sum <= number:
                    queue.append((cur_sum+cur_num, num))
                    queue.append((cur_sum, cur_num*10+num))
        for cur_sum, cur_num in queue:
            if cur_sum + cur_num == number:
                return True
        return False
    
    def punishmentNumber(self, n):
        result = 0
        for number in range(1, n+1):
            if self._checkNum(number):
                result += number ** 2
        return result

#8. String to Integer (atoi)
class MyAtoi:
    def myAtoi(self, s):
        s = s.lstrip().rstrip()
        chars = deque(s)
        if not chars:
            return 0
        negative = chars[0] == '-'
        if negative or chars[0] == '+':
            chars.popleft()
        while chars and chars[0] == '0':
            chars.popleft()
        result = 0
        while chars and chars[0].isdigit():
            result = result * 10 + int(chars.popleft())
            print(result)
            if result > 2 ** 31 - (not negative):
                return (2 ** 31 - (not negative)) * (1 - 2 * negative)
        return result * (1 - 2 * negative)

#7. Reverse Integer
class Reverse:
    def reverse(self, x):
        if x == 0:
            return 0
        num = deque(str(x))
        negative = num[0] == '-'
        if negative:
            num.popleft()
        limit = deque(str(2 ** 31 - (not negative)))
        num.reverse()
        while num[0] == '0':
            num.popleft()
        if len(num) == len(limit):
            for index in range(len(num)):
                if num[index] < limit[index]:
                    break
                elif num[index] > limit[index]:
                    return 0
        return int(''.join(num)) * (1 - 2 * negative)

#1718. Construct the Lexicographically Largest Valid Sequence
class ConstructDistancedSequence:
    def constructDistancedSequence(self, n):
        stack = deque([([None]*(2*n-1), [num for num in range(1,n+1)], 0)])
        while stack:
            cur_arr, cur_nums, new_index = stack.pop()
            while new_index < len(cur_arr) and cur_arr[new_index] is not None:
                new_index += 1
            if new_index == len(cur_arr):
                return cur_arr
            else:
                for num in cur_nums:
                    bias = 0 if num == 1 else num 
                    if (new_index + bias < len(cur_arr)) and (cur_arr[new_index+bias] is None):
                        new_arr, new_nums = list(cur_arr), set(cur_nums)
                        new_arr[new_index], new_arr[new_index+bias] = num, num
                        new_nums.remove(num)
                        stack.append((new_arr, new_nums, new_index))

#1079. Letter Tile Possibilities
class NumTilePossibilities:
    def numTilePossibilities(self, tiles):
        

#2375. Construct Smallest Number From DI String
class SmallestNumber:
    def smallestNumber(self, pattern):   
        res_pat = deque([])
        cur_pat = pattern[0]
        for pat in pattern[1:]:
            if pat == cur_pat[-1]:
                cur_pat += pat
            else:
                res_pat.append(cur_pat)
                cur_pat = pat
        res_pat.append(cur_pat)
        res_pat[0] += res_pat[0][0]
        
        result = []
        numbers = deque(range(1, 10))
        for index, cur_pat in enumerate(res_pat, 1):
            if cur_pat[0] == 'I':
                for _ in range(len(cur_pat)-1):
                    result.append(numbers.popleft())
                if index < len(res_pat):
                    last_num = numbers[len(res_pat[index])]
                    numbers.remove(last_num)
                    result.append(last_num)
                else:
                    result.append(numbers.popleft())
            else:
                part = []
                for _ in range(len(cur_pat)):
                    part.append(numbers.popleft())
                part.reverse()
                result.extend(part)
        return ''.join(map(str, result))

#2563. Count the Number of Fair Pairs
class CountFairPairs:
    def _searchLower(self, nums, target, start_index):
        left_index, right_index = start_index, len(nums) - 1
        if target > nums[right_index]:
            return None
        if target < nums[left_index]:
            return left_index
        while right_index - left_index > 1:
            middle_index = (left_index + right_index) // 2
            if nums[middle_index] >= target:
                right_index = middle_index
            else:
                left_index = middle_index
        return left_index if nums[left_index] == target else right_index
        
    def _searchUpper(self, nums, target, start_index):
        left_index, right_index = start_index, len(nums) - 1
        if target > nums[right_index]:
            return right_index
        if target < nums[left_index]:
            return None
        while right_index - left_index > 1:
            middle_index = (left_index + right_index) // 2
            if nums[middle_index] > target:
                right_index = middle_index
            else:
                left_index = middle_index
        return right_index if nums[right_index] == target else left_index
        
    def countFairPairs(self, nums, lower, upper):
        nums.sort()
        result = 0
        for index, num in enumerate(nums[:-1]):
            if num + nums[-1] < lower:
                continue
            if num + nums[index+1] > upper:
                break
            lower_index = self._searchLower(nums, lower - num, index + 1)
            upper_index = self._searchUpper(nums, upper - num, index + 1)
            if ((lower_index is not None and upper_index is not None)
                and (lower_index <= upper_index)):
                result += upper_index - lower_index + 1
        return result

#1574. Shortest Subarray to be Removed to Make Array Sorted
class FindLengthOfShortestSubarray:
    def _searchEqualOrLess(self, arr, cur_index):
        for index in range(cur_index - 1, -1, -1):
            if arr[index] <= arr[cur_index]:
                return index
        return -1

    def _searchEqualOrGreater(self, arr, cur_index):
        for index in range(cur_index + 1, len(arr)):
            if arr[index] >= arr[cur_index]:
                return index
        return len(arr)
    
    def _checkNonDecreasing(self, arr):
        for index in range(1, len(arr)):
            if arr[index-1] > arr[index]:
                return False
        return True

    def _findLargestSubarray(self, arr, repeat_flag):
        result = 1
        for index in range(1, len(arr)):
            if arr[index-1] > arr[index]:
                if not repeat_flag:
                    return result
                result = 0
            result += 1
        return result
        
    def findLengthOfShortestSubarray(self, arr):
        decreasing_flag = False
        right_target_index = None
        for index in range(1, len(arr)):
            if arr[index-1] > arr[index]:
                if not decreasing_flag:
                    right_target_index = index - 1
                decreasing_flag = True
            elif decreasing_flag:
                left_index = self._searchEqualOrLess(arr, index - 1)
                right_index = self._searchEqualOrGreater(arr, right_target_index)
                if self._checkNonDecreasing(arr[index-1:]):
                    print(index - 1, left_index)
                    print(right_target_index, right_index)
                    return min(index - left_index - 2, right_index - right_target_index - 1)
                break
        return min(len(arr) - self._findLargestSubarray(arr, True),
                   len(arr) - self._findLargestSubarray(arr, False))

#1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence
class IsPrefixOfWord(object):
    def isPrefixOfWord(self, sentence, searchWord):
        string, index = '', 1
        for char in sentence:
            if char == ' ':
                if string[:len(searchWord)] == searchWord:
                    return index
                string = ''
                index += 1
            else:
                string += char
        if string[:len(searchWord)] == searchWord:
            return index
        return -1

#2097. Valid Arrangement of Pairs
class ValidArrangement(object):
    def _buildPath(self, begin_node, graph):
        result = []
        stack = deque([begin_node])
        while stack:
            cur_node = stack[-1]
            if cur_node in graph:
                stack.append(graph[cur_node].pop())
                if not graph[cur_node]:
                    graph.pop(cur_node)
            else:
                result.append(stack.pop())
        result.reverse()
        return result    

    def _buildResult(self, path):
        result = []
        for index in range(1, len(path)):
            result.append([path[index-1], path[index]])
        return result
    
    def validArrangement(self, pairs):
        graph, node_degree = {}, {}
        for begin, end in pairs:
            graph.setdefault(begin, set()).add(end)
            node_degree[begin] = node_degree.setdefault(begin, 0) + 1
            node_degree[end] = node_degree.setdefault(end, 0) - 1

        begin_node = pairs[0][0]
        for node, degree in node_degree.items():
            if degree == 1:
                begin_node = node
                break
            
        path = self._buildPath(begin_node, graph)
        return self._buildResult(path)

#2109. Adding Spaces to a String
class AddSpaces:
    def addSpaces(self, s, spaces):
        res_string_arr = []
        start_index = 0
        spaces.append(len(s))
        for end_index in spaces:
            res_string_arr.append(s[start_index:end_index])
            start_index = end_index
        return ' '.join(res_string_arr)

#3243. Shortest Distance After Road Addition Queries I
class ShortestDistanceAfterQueries:
    def _changeGraph(self, graph, cur_node, prev_node, node_distance):
        if node_distance[cur_node] > node_distance[prev_node] + 1:
            node_distance[cur_node] = node_distance[prev_node] + 1
            if cur_node in graph:
                for next_node in graph[cur_node]:
                    self._changeGraph(graph, next_node, cur_node, node_distance)
            
    def shortestDistanceAfterQueries(self, n, queries):
        graph = {node: {node+1} for node in range(n-1)}
        node_distance = {node: node for node in range(n)}
        result = []
        for start_node, end_node in queries:
            graph[start_node].add(end_node)
            self._changeGraph(graph, end_node, start_node, node_distance)
            result.append(node_distance[n-1])
        return result

class СanMakeSubsequence:
    @functools.cache
    def _getPrevLetter(self, letter):
        return 'z' if letter == 'a' else chr(ord(letter)-1)
        
    def canMakeSubsequence(self, str1, str2):
        str2_index = 0
        for char in str1:
            if ((char == str2[str2_index])
                or (char == self._getPrevLetter(str2[str2_index]))):
                str2_index += 1
            if str2_index == len(str2):
                return True
        return False

#2337. Move Pieces to Obtain a String
class CanChange:  
    def _fillArr(self, arr, string, letter):
        char_amount = 0
        for index, char in enumerate(string):
            char_amount += char == letter
            arr[index][letter] = char_amount
        
    def canChange(self, start, target):
        start_seq = [char for char in start if char!='_']
        target_seq = [char for char in target if char!='_']
        if ''.join(start_seq) != ''.join(target_seq):
            return False
        
        start_index_amount = [{'L': None, 'R': None} for _ in range(len(start))]
        self._fillArr(start_index_amount, start, 'R')
        self._fillArr(start_index_amount[::-1], start[::-1], 'L')
        target_index_amount = [{'L': None, 'R': None} for _ in range(len(target))]
        self._fillArr(target_index_amount, target, 'R')
        self._fillArr(target_index_amount[::-1], target[::-1], 'L')
        for index in range(len(start_index_amount)):
            if ((target_index_amount[index]['L'] > start_index_amount[index]['L'])
                or (target_index_amount[index]['R'] > start_index_amount[index]['R'])):
                return False
        return True  

#2554. Maximum Number of Integers to Choose From a Range I
class MaxCount:
    def maxCount(self, banned, n, maxSum):
        exclude_nums = set(banned)
        result, res_sum = 0, 0
        for num in range(1, n+1):
            if num not in exclude_nums:
                res_sum += num
                if res_sum > maxSum:
                    break
                result += 1
        return result

#2577. Minimum Time to Visit a Cell In a Grid
class MinimumTime:
    def _getNextTime(self, grid, next_node, cur_time):
        target_time = grid[next_node[0]][next_node[1]] - 1
        result = target_time + (target_time - cur_time) % 2 if cur_time < target_time else cur_time
        return result + 1 
        
    def minimumTime(self, grid):
        if (grid[0][1] > 1) and (grid[1][0] > 1):
            return -1
        rows, cols = len(grid), len(grid[0])
        stack = deque([(0, (0, 0))])
        checked_nodes = {(0, 0)}
        while stack:
            cur_time, cur_node = stack.popleft()
            if cur_node == (rows-1, cols-1):
                return cur_time
            for row_inc in range(-1, 2):
                for col_inc in range(-1, 2):
                    if abs(row_inc+col_inc) == 1:
                        next_node = (cur_node[0] + row_inc, cur_node[1] + col_inc)
                        if ((0 <= next_node[0] < rows) 
                            and (0 <= next_node[1] < cols) 
                            and (next_node not in checked_nodes)):
                            next_time = self._getNextTime(grid, next_node, cur_time)
                            bisect.insort(stack, (next_time, next_node))
                            checked_nodes.add(next_node)

#3152. Special Array II
class IsArraySpecial:
    def isArraySpecial(self, nums, queries):
        special_prefix = [0]
        for index, num in enumerate(nums[1:], 1):
            special_prefix.append(special_prefix[-1]+1-(num+nums[index-1])%2)
        result = []
        for begin, end in queries:
            result.append((special_prefix[end]-special_prefix[begin])==0)
        return result

#2981. Find Longest Special Substring That Occurs Thrice I
class MaximumLength:
    def _updateAmount(self, amount, substring):
        for degree in range(1, substring[1]+1):
            new_substring = substring[0] * degree
            amount[new_substring] = amount.setdefault(new_substring, 0) \
                                + substring[1] + 1 - degree
            
    def maximumLength(self, s):
        amount = {}
        prev_substring = [s[0], 1]
        for iter, char in enumerate(s[1:], 1):
            if char == prev_substring[0]:
                prev_substring[1] += 1
            else:
                self._updateAmount(amount, prev_substring)
                prev_substring = [char, 1]
        self._updateAmount(amount, prev_substring)
        result = -1
        for substring, amount in amount.items():
            if amount >= 3:
                result = max(result, len(substring))
        return result

#2779. Maximum Beauty of an Array After Applying Operation
class MaximumBeauty:
    def _searchLimitIndex(self, arr, target):
        left_index, right_index = 0, len(arr) - 1
        if target > arr[-1]:
            return right_index
        while right_index - left_index > 1:
            middle_index = (left_index + right_index) // 2
            if arr[middle_index] <= target:
                left_index = middle_index
            else: 
                right_index = middle_index
        return right_index if arr[right_index] == target else left_index
        
    def maximumBeauty(self, nums, k):
        nums.sort()
        result = 1
        for left_limit, num in enumerate(nums[:-1]):
            target = num + 2 * k
            if ((left_limit + result < len(nums)) 
                and (nums[left_limit+result] <= target)):
                right_limit = left_limit + self._searchLimitIndex(nums[left_limit:], target)
                result = max(result, right_limit - left_limit + 1)
        return result

#2558. Take Gifts From the Richest Pile
class PickGifts:
    def pickGifts(self, gifts, k):
        sorted_gifts = deque(sorted(gifts))
        for _ in range(k):
            max_val = sorted_gifts.pop()
            bisect.insort(sorted_gifts, int(math.sqrt(max_val)))
        return sum(sorted_gifts)

#2290. Minimum Obstacle Removal to Reach Corner
class MinimumObstacles:
    def minimumObstacles(self, grid):
        rows, cols = len(grid), len(grid[0])
        stack = deque([(0, (0, 0))])
        checked_cells = {(0, 0)}
        while stack:
            cur_val, cur_cell = stack.popleft()
            if cur_cell == (rows-1, cols-1):
                return cur_val
            for row_diff in range(-1, 2):
                for col_diff in range(-1, 2):
                    if abs(row_diff+col_diff) == 1:
                        next_cell = (cur_cell[0]+row_diff, cur_cell[1]+col_diff)
                        if ((0 <= next_cell[0] < rows)
                            and (0 <= next_cell[1] < cols)
                            and (next_cell not in checked_cells)):
                            next_val = cur_val if grid[next_cell[0]][next_cell[1]]==0 else cur_val+1
                            bisect.insort(stack, (next_val, next_cell))
                            checked_cells.add(next_cell)

#2593. Find Score of an Array After Marking All Elements
class FindScore:
    def findScore(self, nums):
        num_indexes = {}
        for index, num in enumerate(nums):
            num_indexes.setdefault(num, deque([])).append(index)
        sorted_nums = list(sorted(num_indexes.keys()))
        used_indexes = set()
        result = 0
        for num in sorted_nums:
            while num_indexes[num]:
                index = num_indexes[num].popleft()
                if index not in used_indexes:
                    for used_index in range(index-1, index+2):
                        if 0 <= used_index < len(nums):
                            used_indexes.add(used_index)
                    result += num
        return result

#2054. Two Best Non-Overlapping Events
class MaxTwoEvents:
    def _searchStart(self, postfix, target):
        left_index, right_index = 0, len(postfix) - 1
        if target > postfix[right_index][0]:
            return 0
        while right_index - left_index > 1:
            if postfix[left_index] == target:
                return postfix[left_index][1]
            if postfix[right_index] == target:
                return postfix[right_index][1]
            middle_index = (left_index + right_index) // 2
            if postfix[middle_index][0] < target:
                left_index = middle_index
            else:
                right_index = middle_index
        return postfix[right_index][1]
        
    def maxTwoEvents(self, events):
        start_value = {}
        for start, end, value in events:
            start_value[start] = max(start_value.setdefault(start, 0), value)
        postfix = deque([])
        for start in sorted(start_value.keys(), reverse=True):
            if not postfix:
                postfix.appendleft((start, start_value[start]))
            else:
                postfix.appendleft((start, max(postfix[0][1], start_value[start])))
        result = postfix[0][1]
        for start, end, value in events:
            full_value = value + self._searchStart(postfix, end+1)
            result = max(result, full_value)
        return result

#3264. Final Array State After K Multiplication Operations I
class GetFinalState:
    def getFinalState(self, nums, k, multiplier):
        nums_index = deque(sorted([(num, index) for index, num in enumerate(nums)]))
        for _ in range(k):
            cur_num, cur_index = nums_index.popleft()
            bisect.insort(nums_index, (cur_num*multiplier, cur_index))
        result = [None for _ in range(len(nums_index))]
        for num, index in nums_index:
            result[index] = num
        return result

#2182. Construct String With Repeat Limit
class RepeatLimitedString:
    def repeatLimitedString(self, s, repeatLimit):
        letters = list(s)
        letters.sort(key=ord, reverse=True)
        replace_index, repeats = 1, 1
        for index in range(1, len(letters)):
            if letters[index] == letters[index-1]:
                repeats += 1
            else:
                repeats = 1
            if repeats > repeatLimit:
                try:
                    while letters[replace_index] == letters[index]:
                        replace_index += 1
                    letters[index], letters[replace_index] = letters[replace_index], letters[index]
                    replace_index += 1
                except IndexError:
                    return ''.join(letters[:index])
            if replace_index == index:
                replace_index += 1
        return ''.join(letters)

#1475. Final Prices With a Special Discount in a Shop
class FinalPrices:
    def finalPrices(self, prices):
        queue = deque([(prices[0], 0)])
        result = [None for _ in range(len(prices))]
        for index, price in enumerate(prices[1:], 1):
            while queue and price <= queue[-1][0]:
                cur_price, cur_index = queue.pop()
                result[cur_index] = cur_price - price 
            queue.append((price, index))
        while queue:
            cur_price, cur_index = queue.pop()
            result[cur_index] = cur_price
        return result

#1792. Maximum Average Pass Ratio
class MaxAverageRatio:
    def _getIncrease(self, class_info):
        num, delim = class_info
        return (delim-num) / (delim*(delim+1))
        
    def maxAverageRatio(self, classes, extraStudents):
        sorted_classes = deque(sorted(classes, key=self._getIncrease))
        for _ in range(extraStudents):
            cur_pass, cur_total = sorted_classes.pop()
            bisect.insort(sorted_classes, [cur_pass+1, cur_total+1], key=self._getIncrease)
        result = functools.reduce(lambda x,y: x+y[0]/y[1], sorted_classes, 0) / len(sorted_classes)
        return result

#769. Max Chunks To Make Sorted
class MaxChunksToSorted:
    def maxChunksToSorted(self, arr):
        indexes, nums = set(), set()
        result = 0
        for index, num in enumerate(arr):
            indexes.add(index)
            nums.add(num)
            if not indexes ^ nums:
                result += 1
                indexes, nums = set(), set()
        return result

#2762. Continuous Subarrays
class ContinuousSubarrays:
    def continuousSubarrays(self, nums):
        max_min = deque([])
        left_index = 0
        result = 0
        for right_index, num in enumerate(nums):
            bisect.insort(max_min, (num, right_index))
            if max_min[-1][0] - max_min[0][0] > 2:
                while max_min[-1][0] - max_min[0][0] > 2:
                    if max_min[-1][1] > max_min[0][1]:
                        left_index = max(left_index, max_min.popleft()[1])
                    else:
                        left_index = max(left_index, max_min.pop()[1])
                while max_min[-1][1] < left_index:
                    max_min.pop()
                while max_min[0][1] < left_index:
                    max_min.popleft()
                left_index += 1
            result += right_index - left_index + 1
        return result

#1346. Check If N and Its Double Exist
class CheckIfExist:
    def checkIfExist(self, arr):
        checked_nums = set()
        for num in arr:
            if (num*2 in checked_nums) or ((num%2==0) and (num/2 in checked_nums)):
                return True
            checked_nums.add(num)
        return False

#2415. Reverse Odd Levels of Binary Tree
class ReverseOddLevels:
    def reverseOddLevels(self, root):
        queue = deque([root])
        reversed_vals = deque([])
        lvl = 0
        while queue:
            temp_queue = deque([])
            while queue:
                cur_node = queue.popleft()
                if lvl % 2 == 1:
                    cur_node.val = reversed_vals.popleft()
                else:
                    if cur_node.right:
                        reversed_vals.appendleft(cur_node.right.val)
                        reversed_vals.appendleft(cur_node.left.val)
                if cur_node.right:
                    temp_queue.append(cur_node.right)
                    temp_queue.append(cur_node.left)
            queue = temp_queue
            lvl += 1
        return root

#2471. Minimum Number of Operations to Sort a Binary Tree by Level
class MinimumOperations:
    def _getValNodes(self, root, val_nodes):
        val_nodes[root.val] = root
        if root.left:
            self._getValNodes(root.left, val_nodes)
        if root.right:
            self._getValNodes(root.right, val_nodes)
        
    def minimumOperations(self, root):
        val_nodes = {}
        self._getValNodes(root, val_nodes)
        queue = deque([root])
        sorted_vals = deque([root.val])
        result = 0
        while queue:
            temp_queue = deque([])
            temp_sorted_vals = deque([])
            while queue:
                cur_node = queue.popleft()
                cur_sorted_val = sorted_vals.popleft()
                if cur_node.val != cur_sorted_val:
                    val_nodes[cur_node.val], val_nodes[cur_sorted_val] = (val_nodes[cur_sorted_val],
                                                                          val_nodes[cur_node.val])
                    val_nodes[cur_node.val].val = cur_node.val
                    cur_node.val = cur_sorted_val
                    result += 1
                if cur_node.left:
                    bisect.insort(temp_sorted_vals, cur_node.left.val)
                    temp_queue.append(cur_node.left)
                if cur_node.right:
                    bisect.insort(temp_sorted_vals, cur_node.right.val)
                    temp_queue.append(cur_node.right)
            queue = temp_queue
            sorted_vals = temp_sorted_vals
        return result

#3203. Find Minimum Diameter After Merging Two Trees
class MinimumDiameterAfterMerge:
    def _getTree(self, edges):
        tree = {}
        for start, end in edges:
            tree.setdefault(start, set()).add(end)
            tree.setdefault(end, set()).add(start)
        return tree

    def _getTreeDiameter(self, tree):
        result = 0
        leafs = deque(node for node in tree if len(tree[node]) == 1)
        while len(tree) > 2:
            for _ in range(len(leafs)):
                leaf = leafs.popleft()
                parent = tree[leaf].pop()
                tree.pop(leaf)
                tree[parent].remove(leaf)
                if len(tree[parent]) == 1:
                    leafs.append(parent)
            result += 1 
        return result + len(tree) - 1
        
    def minimumDiameterAfterMerge(self, edges1, edges2):
        diam_1, diam_2 = 0, 0
        roots_1, roots_2 = 0, 0
        if edges1:
            tree_1 = self._getTree(edges1)
            diam_1 = self._getTreeDiameter(tree_1)
            roots_1 += len(tree_1) - 1
        if edges2:
            tree_2 = self._getTree(edges2)
            diam_2 = self._getTreeDiameter(tree_2)
            roots_2 += len(tree_2) - 1
        return max(diam_1+diam_2+1, 2*diam_1-roots_1, 2*diam_2-roots_2)

#14. Longest Common Prefix
class LongestCommonPrefix:
    def longestCommonPrefix(self, strs):
        cur_index = 0
        while True:
            for string in strs:
                if ((len(string) == cur_index) 
                    or (strs[0][cur_index] != string[cur_index])):
                    return strs[0][:cur_index]
            cur_index += 1

#1014. Best Sightseeing Pair
class MaxScoreSightseeingPair:
    def _getPrefix(self, values):
        result = deque([])
        for index, num in list(enumerate(values))[:0:-1]:
            sum_val = num + values[0] - index
            if not result:
                result.appendleft(sum_val)
            else:
                result.appendleft(max(result[0], sum_val))
        return result
        
    def maxScoreSightseeingPair(self, values):
        prefix = self._getPrefix(values)
        result = prefix[0]
        for index, num in enumerate(values[1:-1], 1):
            diff = num - values[0] + index
            result = max(result, prefix[index]+diff)
        return result

#1765. Map of Highest Peak
class HighestPeak:
    def highestPeak(self, isWater):
        height_matrix = [[-1] * len(isWater[0]) for _ in range(len(isWater))]
        queue = set((row, col) for row in range(len(isWater)) for col in range(len(isWater[0]))
                               if isWater[row][col]==1)
        checked_cells = set()
        while queue:
            temp_queue = set()
            checked_cells.update(queue)
            while queue:
                cur_row, cur_col = queue.pop()
                height_matrix[cur_row][cur_col] += 1
                for row_diff in range(-1, 2):
                    for col_diff in range(-1, 2):
                        if abs(row_diff+col_diff) == 1:
                            next_row = cur_row + row_diff
                            next_col = cur_col + col_diff
                            if ((0 <= next_row < len(isWater))
                                and (0 <= next_col < len(isWater[0]))
                                and (next_row, next_col) not in checked_cells):
                                height_matrix[next_row][next_col] = (height_matrix[cur_row][cur_col] if 
                                    height_matrix[next_row][next_col]==-1 else min(height_matrix[cur_row][cur_col],
                                                                                   height_matrix[next_row][next_col]))
                                temp_queue.add((next_row, next_col))
            queue = temp_queue
        return height_matrix

#Count Servers that Communicate
class CountServers:
    def countServers(self, grid):
        single_on_row, single_on_col = set(), set()
        once_amount = 0
        for row in range(len(grid)):
            candidate = None
            temp_once_amount = 0
            for col in range(len(grid[0])):
                if grid[row][col] == 1:
                    candidate = (row, col)
                    temp_once_amount += 1
            if temp_once_amount > 1:
                candidate = None
            if candidate is not None:
                single_on_row.add((candidate))
            once_amount += temp_once_amount
            
        for col in range(len(grid[0])):
            candidate = None
            for row in range(len(grid)):
                if grid[row][col] == 1:
                    if candidate is not None:
                        candidate = None
                        break
                    candidate = (row, col)
            if candidate is not None:
                single_on_col.add((candidate))
        return once_amount - len(single_on_row & single_on_col)

#1462. Course Schedule IV
class CheckIfPrerequisite:    
    def checkIfPrerequisite(self, numCourses, prerequisites, queries):
        reversed_graph, node_degree, leafs = {}, {}, set(node for node in range(numCourses))
        for begin, end in prerequisites:
            reversed_graph.setdefault(end, set()).add(begin)
            node_degree[begin] = node_degree.setdefault(begin, 0) + 1
            if begin in leafs:
                leafs.remove(begin)
        leafs = deque(leafs)
        
        node_routes = defaultdict(set)
        while leafs:
            for _ in range(len(leafs)):
                cur_node = leafs.popleft()
                for prev_node in reversed_graph.get(cur_node, []):
                    node_routes[prev_node].update(node_routes[cur_node].union({cur_node}))
                    node_degree[prev_node] -= 1
                    if node_degree[prev_node] == 0:
                        leafs.append(prev_node)

        result = [bool(node_routes[begin]&{end}) for begin, end in queries]
        return result

#802. Find Eventual Safe States
class EventualSafeNodes:
    def eventualSafeNodes(self, graph):
        reversed_graph, node_degree, leafs = {}, {}, deque([])
        for begin_node, end_nodes in enumerate(graph):
            for end_node in end_nodes:
                reversed_graph.setdefault(end_node, set()).add(begin_node)
                node_degree[begin_node] = node_degree.setdefault(begin_node, 0) + 1
            if not end_nodes:
                leafs.append(begin_node)

        result = []
        while leafs:
            for _ in range(len(leafs)):
                cur_node = leafs.popleft()
                for prev_node in reversed_graph.get(cur_node, []):
                    node_degree[prev_node] -= 1
                    if node_degree[prev_node] == 0:
                        leafs.append(prev_node)
                result.append(cur_node)
        result.sort()
        return result

#2658. Maximum Number of Fish in a Grid
class FindMaxFish:
    def _catchAllFish(self, grid, start_row, start_col):
        result = 0
        stack = deque([(start_row, start_col)])
        while stack:
            row, col = stack.pop()
            result += grid[row][col]
            grid[row][col] = 0
            for row_diff in range(-1, 2):
                for col_diff in range(-1, 2):
                    if abs(row_diff+col_diff) == 1:
                        next_row, next_col = row + row_diff, col + col_diff
                        if ((0 <= next_row < len(grid)) 
                            and (0 <= next_col < len(grid[0]))
                            and (grid[next_row][next_col] != 0)):
                            stack.append((next_row, next_col))
        return result   
    
    def findMaxFish(self, grid):
        result = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != 0:
                    result = max(result, self._catchAllFish(grid, row, col))
        return result

#2948. Make Lexicographically Smallest Array by Swapping Elements
class LexicographicallySmallestArray:
    def lexicographicallySmallestArray(self, nums, limit):
        index_nums = list((num, index) for index, num in enumerate(nums))
        index_nums.sort()
        group = 0
        groups = {group: deque([index_nums[0][0]])}
        index_group = {index_nums[0][1]: group}
        for index in range(1, len(index_nums)):
            if index_nums[index][0] - index_nums[index-1][0] > limit:
                group += 1
                groups[group] = deque([])
            groups[group].append(index_nums[index][0])
            index_group[index_nums[index][1]] = group

        result = []
        for index in range(len(index_nums)):
            result.append(groups[index_group[index]].popleft())
        return result

#684. Redundant Connection
class FindRedundantConnection:
    def _getAnswear(self, start_node, visited_from, edges):
        next_node = visited_from[start_node]
        cycle_edges = {(start_node, next_node)}
        while True:
            if next_node == start_node:
                break
            cur_node = next_node
            next_node = visited_from[next_node]
            cycle_edges.add((cur_node, next_node))
        for edge in edges[::-1]:
            if ((edge[0], edge[1]) in cycle_edges
                or (edge[1], edge[0]) in cycle_edges):
                return edge
    
    def findRedundantConnection(self, edges):
        visited, visited_from = [0] * (len(edges) + 1), [-1] * (len(edges) + 1)
        graph = {}
        for node_1, node_2 in edges:
            graph.setdefault(node_1, set()).add(node_2)
            graph.setdefault(node_2, set()).add(node_1)
            
        stack = deque([1])
        while stack:
            cur_node = stack[-1]
            visited[cur_node] = 1
            leaf_flag = True
            for next_node in graph[cur_node]:
                if ((visited_from[cur_node] != next_node)
                    and (visited[next_node] != 2)):
                    visited_from[next_node] = cur_node
                    if visited[next_node] == 1:
                        return self._getAnswear(next_node, visited_from, edges)
                    stack.append(next_node)
                    leaf_flag = False
            if leaf_flag:
                visited[cur_node] =  2
                stack.pop()

#2017. Grid Game
class GridGame:
    def gridGame(self, grid):
        prefix = deque([deque([grid[0][0]]), deque([grid[1][-1]])])
        for index in range(1, len(grid[0])):
            prefix[0].append(prefix[0][-1]+grid[0][index])
            prefix[1].appendleft(prefix[1][0]+grid[1][-index-1])

        result = float('inf')
        for index in range(len(grid[0])):
            min_sum = max(prefix[0][-1]-prefix[0][index],
                          prefix[1][0]-prefix[1][index])
            result = min(result, min_sum)
        return result

#2661. First Completely Painted Row or Column
class FirstCompleteIndex:
    def firstCompleteIndex(self, arr, mat):
        matrix = {}
        for row in range(len(mat)):
            for col in range(len(mat[0])):
                matrix[mat[row][col]] = (row, col)
        rows, cols = [len(mat[0])] * len(mat), [len(mat)] * len(mat[0])
        for index, num in enumerate(arr):
            row, col = matrix[num]
            rows[row] -= 1
            if rows[row] == 0:
                return index
            cols[col] -= 1
            if cols[col] == 0:
                return index

#2425. Bitwise XOR of All Pairings
class XorAllNums:
    def _xorPower(self, num, power):
        return 0 if power % 2 == 0 else 0 ^ num
        
    def xorAllNums(self, nums1, nums2):
        xor_nums_1 = functools.reduce(lambda x, y: x ^ self._xorPower(y, len(nums2)), 
                                      nums1[1:], 
                                      self._xorPower(nums1[0], len(nums2)))
        xor_nums_2 = functools.reduce(lambda x, y: x ^ self._xorPower(y, len(nums1)), 
                                      nums2[1:], 
                                      self._xorPower(nums2[0], len(nums1)))
        return xor_nums_1 ^ xor_nums_2

#1800. Maximum Ascending Subarray Sum
class MaxAscendingSum:
    def maxAscendingSum(self, nums):
        result = 0
        temp_sum = nums[0]
        for index, num in enumerate(nums[1:], 1):
            if num > nums[index-1]:
                temp_sum += num
            else:
                result = max(result, temp_sum)
                temp_sum = num
        return max(result, temp_sum)

#1790. Check if One String Swap Can Make Strings Equal
class AreAlmostEqual:
    def areAlmostEqual(self, s1, s2):
        mismatched_indexes = []
        for index in range(len(s1)):
            if s1[index] != s2[index]:
                mismatched_indexes.append(index)
            if len(mismatched_indexes) > 2:
                return False
        return not mismatched_indexes or (len(mismatched_indexes) == 2
                                          and (s1[mismatched_indexes[0]] == s2[mismatched_indexes[1]])
                                          and (s1[mismatched_indexes[1]] == s2[mismatched_indexes[0]]))

#1752. Check if Array Is Sorted and Rotated
class Check:
    def _findBorder(self, nums):
        for index, num in enumerate(nums[1:], 1):
            if num < nums[index-1]:
                return index
        return 0
            
    def check(self, nums):
        border = self._findBorder(nums)
        for index in range(1, len(nums)):
            num_index = (index + border) % len(nums)
            if nums[num_index] < nums[num_index-1]:
                return False
        return True

#3151. Special Array I
class IsArraySpecial:
    def isArraySpecial(self, nums):
        nums[0] %= 2
        for index in range(1, len(nums)):
            nums[index] %= 2
            if nums[index] == nums[index-1]:
                return False
        return True

#1930. Unique Length-3 Palindromic Subsequences
class CountPalindromicSubsequence:
    def _refreshRight(self, letter, letters, amount):
        amount[letter] -= 1
        if amount[letter] == 0:
            letters.remove(letter)

    def _refreshLeft(self, letter, letters, amount):
        amount[letter] = amount.setdefault(letter, 0) + 1
        if amount[letter] == 1:
            letters.add(letter)
        
    def countPalindromicSubsequence(self, s):
        left_letters, right_letters = {s[0]}, set()
        left_amount, right_amount = {s[0]: 1}, {}
        result = {}
        for letter in s[1:]:
            right_letters.add(letter)
            right_amount[letter] = right_amount.setdefault(letter, 0) + 1
        for center in s[1:-1]:
            self._refreshRight(center, right_letters, right_amount)
            for tails in left_letters & right_letters:
                result[center][tails] = result.setdefault(center, dict()).setdefault(tails, 1)
            self._refreshLeft(center, left_letters, left_amount)
        return functools.reduce(lambda x, y: x+sum(y.values()), result.values(), 0)

#1726. Tuple with Same Product
class TupleSameProduct:
    def tupleSameProduct(self, nums):
        product_dict = {}
        result = 0
        for index, num_1 in enumerate(nums[:-1], 1):
            for num_2 in nums[index:]:
                product = num_1 * num_2
                product_dict[product] = product_dict.setdefault(product, 0) + 1
                result += 8 * (product_dict[product] - 1)
        return result

#3160. Find the Number of Distinct Colors Among the Balls
class QueryResults:
    def queryResults(self, limit, queries):
        color_ball, ball_color = {}, {}
        result = []
        for ball, color in queries:
            if ball in ball_color:
                color_ball[ball_color[ball]].remove(ball)
                if not color_ball[ball_color[ball]]:
                    color_ball.pop(ball_color[ball])
            ball_color[ball] = color
            color_ball.setdefault(color, set()).add(ball)
            result.append(len(color_ball))
        return result

#827. Making A Large Island
class LargestIsland:
    def _getIslandSizeAndBorder(self, start_row, start_col, grid, island_no):
        border = set()
        size = 0
        stack = deque([(start_row, start_col)])
        while stack:
            cur_row, cur_col = stack.pop()
            if grid[cur_row][cur_col] == 1:
                size += 1
                for row_diff in range(-1, 2):
                    for col_diff in range(-1, 2):
                        if abs(row_diff+col_diff) == 1:
                            next_row = cur_row + row_diff
                            next_col = cur_col + col_diff
                            if ((0 <= next_row < len(grid))
                                and (0 <= next_col < len(grid[0]))
                                and (grid[next_row][next_col] != island_no)):
                                stack.append((next_row, next_col))
                grid[cur_row][cur_col] = 2
            elif grid[cur_row][cur_col] <= 0:
                border.add((cur_row, cur_col))
                grid[cur_row][cur_col] = island_no
        return size, border
        
    def largestIsland(self, grid):
        water_only = True
        islands, island_no = [], -1
        for start_row in range(len(grid)):
            for start_col in range(len(grid[0])):
                if grid[start_row][start_col] == 1:
                    size, border = self._getIslandSizeAndBorder(start_row, start_col, grid, island_no)
                    islands.append((size, border))
                    island_no -= 1
                    water_only = False

        if water_only:
            return 1
        elif (len(islands) == 1) and not islands[0][1]:
            return islands[0][0]
        else:
            common_borders = {}
            for size, border in islands:
                for cell in border:
                    common_borders[cell] = common_borders.setdefault(cell, 0) + size
            return max(common_borders.values()) + 1

#3066. Minimum Operations to Exceed Threshold Value II
class MinOperations:
    def _complete(self, arr_from, arr_into, target):
        return min(arr_from[0], arr_into[0] if arr_into else arr_from[0]) >= target

    def _getMinNum(self, arr_from, arr_into):
        if arr_into and arr_into[0] < arr_from[0]:
            return arr_into.popleft()
        else:
            return arr_from.popleft()
        
    def minOperations(self, nums, k):
        arr_from = deque(sorted(nums))
        arr_into = deque([])
        result = 0
        while not self._complete(arr_from, arr_into, k):
            min_num = self._getMinNum(arr_from, arr_into)
            if not arr_from:
                arr_from, arr_into = arr_into, arr_from
            max_num = self._getMinNum(arr_from, arr_into)
            arr_into.append(max_num+min_num*2)
            if not arr_from:
                arr_from, arr_into = arr_into, arr_from
            result += 1
        return result

#1352. Product of the Last K Numbers
class ProductOfNumbers:
    def __init__(self):
        self.prefix_product = deque([1])

    def add(self, num):
        if num == 0:
            self.prefix_product = deque([1])
        else:
            self.prefix_product.append(self.prefix_product[-1]*num)

    def getProduct(self, k):
        if k >= len(self.prefix_product):
            return 0
        else:
            return self.prefix_product[-1] // self.prefix_product[-k-1]

#1415. The k-th Lexicographical String of All Happy Strings of Length n
class GetHappyString:
    def getHappyString(self, n, k):
        strings = deque([''])
        letters = ['a', 'b', 'c']
        for _ in range(n):
            for _ in range(len(strings)):
                string = strings.popleft()
                for letter in letters:
                    if (string == '') or (string[-1] != letter):
                        strings.append(string+letter)
        return '' if k-1 >= len(strings) else strings[k-1]

#1980. Find Unique Binary String
class FindDifferentBinaryString:
    def _getNumberOrder(self, numbers):
        items = list(numbers.items())
        if len(items) == 1:
            if items[0] == '1':
                return ['1', '0']
            else:
                return ['0', '1']
        if items[0][1] < items[1][1]:
            return [items[1][0], items[0][0]]
        else:
            return [items[0][0], items[1][0]]
        
    def findDifferentBinaryString(self, nums):
        positions = [dict() for _ in range(len(nums))]
        for num in nums:
            for index, number in enumerate(num):
                positions[index][number] = positions[index].setdefault(number, 0) + 1

        nums_set = set(nums)
        stack = deque([''])
        while stack:
            cur_string = stack.pop()
            cur_index = len(cur_string)
            if (cur_index == len(nums)) and (cur_string not in nums_set):
                return cur_string
            elif cur_index < len(nums):
                for number in self._getNumberOrder(positions[cur_index]):
                    stack.append(cur_string+number)

#1261. Find Elements in a Contaminated Binary Tree
class FindElements(object):
    def _goTree(self, node, node_val):
        node.val = node_val
        if node.left:
            yield from self._goTree(node.left, 2*node.val+1)
        if node.right:
            yield from self._goTree(node.right, 2*node.val+2)
        yield node.val
    
    def __init__(self, root):
        self.node_vals = set(self._goTree(root, 0))

    def find(self, target):
        return target in self.node_vals

#1028. Recover a Tree From Preorder Traversal
class RecoverFromPreorder:
    def recoverFromPreorder(self, traversal):
        nodes = deque(map(int, re.split(r'-{1,}', traversal)))
        lvls = deque(map(len, re.split(r'\d{1,}', traversal)[1:-1]))
        lvl_node = {0: [TreeNode(nodes.popleft())]}
        while lvls and nodes:
            lvl = lvls.popleft()
            child = TreeNode(nodes.popleft())
            lvl_node.setdefault(lvl, list()).append(child)
            parent = lvl_node[lvl-1][-1]
            if not parent.left:
                parent.left = child
            else:
                parent.right = child
        return lvl_node[0][0]

#889. Construct Binary Tree from Preorder and Postorder Traversal
class ConstructFromPrePost:
    def constructFromPrePost(self, preorder, postorder):
        node_order = [None] * (len(postorder) + 1)
        for index, node_val in enumerate(postorder):
            node_order[node_val] = index
        lvl_nodes = {0: [TreeNode(preorder[0])]}
        lvl = 1
        for node_val in preorder[1:]:
            while (((lvl_nodes[lvl-1][-1].left) and (lvl_nodes[lvl-1][-1].right))
                   or (node_order[lvl_nodes[lvl-1][-1].val] < node_order[node_val])):
                lvl -= 1
            node = TreeNode(node_val)
            if not lvl_nodes[lvl-1][-1].left:
                lvl_nodes[lvl-1][-1].left = node
            else:
                lvl_nodes[lvl-1][-1].right = node
            lvl_nodes.setdefault(lvl, list()).append(node)
            lvl += 1
        return lvl_nodes[0][0]

#1524. Number of Sub-arrays With Odd Sum
class NumOfSubarrays:
    def numOfSubarrays(self, arr):
        result = deque([{'valid': arr[-1]%2, 'invalid': (arr[-1]+1)%2}])
        for num in reversed(arr[:-1]):
            if num % 2 == 1:
                result.appendleft({'valid': result[0]['invalid']+1,
                                   'invalid': result[0]['valid']})
            else:
                result.appendleft({'valid': result[0]['valid'],
                                   'invalid': result[0]['invalid']+1})
        return functools.reduce(lambda x, y: x+y['valid'], result, 0) % (10 ** 9 + 7)

#1079. Letter Tile Possibilities
class NumTilePossibilities(object):
    def numTilePossibilities(self, tiles):
        queue = deque([('', list(tiles))])
        result = set()
        for _ in range(len(tiles)+1):
            for _ in range(len(queue)):
                cur_string, cur_letters = queue.popleft()
                if cur_string:
                    result.add(cur_string)
                if cur_letters:
                    for index in range(len(cur_letters)):
                        new_letters = list(cur_letters)
                        new_string = cur_string + new_letters.pop(index)
                        queue.append((new_string, new_letters))
        return len(result)

#19. Remove Nth Node From End of List
class RemoveNthFromEnd:
    def removeNthFromEnd(self, head, n):
        last_nodes = deque([None])
        cur_node = head
        while cur_node:
            last_nodes.append(cur_node)
            if len(last_nodes) > n + 1:
                last_nodes.popleft()
            cur_node = cur_node.next
        last_nodes.append(None)

        prev_node, next_node = last_nodes[-(n+2)], last_nodes[-n]
        if prev_node is not None:
            prev_node.next = next_node
        else:
            head = last_nodes[-n]
        return head

#1749. Maximum Absolute Sum of Any Subarray
class MaxAbsoluteSum:
    def maxAbsoluteSum(self, nums):
        result = abs(nums[-1])
        memory = deque([{'max': nums[-1], 'min': nums[-1]}])
        for num in nums[-2::-1]:
            next_memory = {'max': max(memory[0]['max']+num, num),
                           'min': min(memory[0]['min']+num, num)}
            result = max(result, abs(next_memory['max']), abs(next_memory['min']))
            memory.appendleft(next_memory)
        return result

#873. Length of Longest Fibonacci Subsequence
class LenLongestFibSubseq:
    def _findSubseqLength(self, num_i_2, num_i_1, nums):
        num_sum = num_i_2 + num_i_1
        if num_sum in nums:
            return 1 + self._findSubseqLength(num_i_1, num_sum, nums)
        return 3
        
    def lenLongestFibSubseq(self, arr):
        nums = set(arr)
        result = 0
        for index, num_1 in enumerate(arr[:-2], 1):
            for num_2 in arr[index:-1]:
                num_sum = num_1 + num_2
                if num_sum > arr[-1]:
                    break
                elif num_sum in nums:
                    result = max(result, self._findSubseqLength(num_2, num_sum, nums))
        return result

#1371. Find the Longest Substring Containing Vowels in Even Counts
class FindTheLongestSubstring:
    def findTheLongestSubstring(self, s):
        substrings = [{}]
        vowels = {'a', 'e', 'i', 'o', 'u'}
        first_index = {'': 0}
        result = 0
        for index, letter in enumerate(s, 1):
            next_substring = dict(substrings[-1])
            if letter in vowels:
                next_substring[letter] = next_substring.setdefault(letter, 0) + 1
                if next_substring[letter] % 2 == 0:
                    next_substring.pop(letter)
            substrings.append(next_substring)
            search_string = ''
            for search_letter in sorted(substrings[-1]):
                search_string += search_letter + '1'
            result = max(result, index-first_index.setdefault(search_string, index))
        return result

#1760. Minimum Limit of Balls in a Bag
class MinimumSize:
    def minimumSize(self, nums, maxOperations):
        left, right = 1, max(nums)
        while right - 1 > left:
            max_balls = (right + left) // 2
            cur_operations = functools.reduce(lambda x, y: x+(y-1)//max_balls, nums, 0)
            if cur_operations > maxOperations:
                left = max_balls
            else:
                right = max_balls
        return left if functools.reduce(lambda x, y: x+(y-1)//left, nums, 0) <= maxOperations else right

#2460. Apply Operations to an Array
class ApplyOperations:
    def applyOperations(self, nums):
        zeros = 0
        non_zero = []
        for index in range(len(nums)):
            if (index < len(nums) - 1) and (nums[index] == nums[index+1]):
                nums[index] *= 2
                nums[index+1] = 0
            if nums[index] != 0:
                non_zero.append(nums[index])
            else:
                zeros += 1
        for _ in range(zeros):
            non_zero.append(0)
        return non_zero

##1813. Sentence Similarity III
class AreSentencesSimilar:
    def _checkBegining(self, strings, substrings):
        index = 0
        while index < len(substrings):
            if strings[index] != substrings[index]:
                break
            index += 1
        return index

    def _checkEnding(self, strings, substrings):
        index = len(substrings)
        while index > 0:
            if strings[-index] != substrings[-index]:
                break
            index -= 1
        return index
        
    def areSentencesSimilar(self, sentence1, sentence2):
        arr_1, arr_2 = sentence1.split(' '), sentence2.split(' ')
        long_arr = max(arr_1, arr_2, key=len)
        short_arr = arr_1 if long_arr == arr_2 else arr_2
        begining_part = self._checkBegining(long_arr, short_arr)
        ending_part = self._checkEnding(long_arr, short_arr[begining_part:])
        return True if ending_part == 0 else False

##1106. Parsing A Boolean Expression
class ParseBoolExpr:
    def _checkValues(self, operations, value):
        for index, val in enumerate(operations[value]):
            if val is not True and val is not False:
                operations[value][index] = operations[val]
    
    def _makeOperations(self, operation, values):
        if operation == '!':
            return not values[0]
        elif operation == '&':
            return functools.reduce(lambda x, y: x and y, values[1:], values[0])
        elif operation == '|':
            return functools.reduce(lambda x, y: x or y, values[1:], values[0])
        
    def parseBoolExpr(self, expression):
        operations_icons = {'!', '&', '|'}
        value_icons = {'t': True, 'f': False}
        parentheses_icons = {'(': 1, ')': -1}
        operations = {}
        operations_lvl = {}
        lvl = 0
        if len(expression) == 1:
            return value_icons[expression[0]]
        for index, char in enumerate(expression):
            if char in operations_icons:
                operations[index] = list()
                if lvl > 0:
                    operations[operations_lvl[lvl-1][-1]].append(index)
                operations_lvl.setdefault(lvl, list()).append(index)
            elif char in parentheses_icons:
                lvl += parentheses_icons[char]
            elif char in value_icons:
                operations[operations_lvl[lvl-1][-1]].append(value_icons[char])
                
        for key in sorted(operations_lvl, reverse=True):
            for value in operations_lvl[key]:
                self._checkValues(operations, value)
                operations[value] = self._makeOperations(expression[value], operations[value])
        return operations[0]

#2570. Merge Two 2D Arrays by Summing Values
class MergeArrays:
    def mergeArrays(self, nums1, nums2):
        for nums in nums2:
            insert_index = bisect.bisect(nums1, nums)
            if (insert_index > 0) and (nums[0] == nums1[insert_index-1][0]):
                nums1[insert_index-1][1] += nums[1]
            elif (insert_index < len(nums1)) and (nums[0] == nums1[insert_index][0]):
                nums1[insert_index][1] += nums[1]
            else:
                bisect.insort(nums1, nums)
        return nums1

#2161. Partition Array According to Given Pivot
class PivotArray:
    def pivotArray(self, nums, pivot):
        less, equal = 0, 0
        for num in nums:
            less += num < pivot
            equal += num == pivot
        index_type = {-1: 0, 0: less, 1: less+equal}
        result = [None] * len(nums)
        for num in nums:
            diff = num - pivot
            index = 0 if diff == 0 else diff // abs(diff)
            result[index_type[index]] = num
            index_type[index] += 1
        return result

#241. Different Ways to Add Parentheses
class DiffWaysToCompute:
    def _parenthesesDegree(self, operation):
        char_dict = {'(': 1, ')': -1}
        return functools.reduce(lambda x, y: x + char_dict[y] if y in char_dict else 0, operation, 0)
        
    def _getParenthesesString(self, seq):
        seq_dict = {int(operation): index for index, operation in enumerate(seq)}
        for operation in range(len(seq_dict)):
            oper_index = seq_dict[operation]
            degree = self._parenthesesDegree(seq[oper_index+1])
            while (oper_index < len(seq) - 2) and (degree != 0):
                oper_index += 1
                degree += self._parenthesesDegree(seq[oper_index+1])
            seq[oper_index] += ')'

            while seq[oper_index-1]:
                pass
            seq[oper_index] += '('
        
    def diffWaysToCompute(self, expression):
        stack = deque([([], {'0','1','2'})])
        result = []
        while stack:
            cur_seq, cur_operations = stack.pop()
            if not cur_operations:
                result.append(cur_seq)
            for operation in cur_operations:
                next_seq, next_operations = list(cur_seq), set(cur_operations)
                next_seq.append(operation)
                next_operations.remove(operation)
                stack.append((next_seq, next_operations))
        return result

#1780. Check if Number is a Sum of Powers of Three
class CheckPowersOfThree:
    def _getDegree(self, num):
        degree = -1
        while num > 0:
            num //= 3
            degree += 1
        return degree
        
    def checkPowersOfThree(self, n):
        checked_powers = set()
        while n > 0:
            power = self._getDegree(n)
            if power in checked_powers:
                break
            checked_powers.add(power)
            n -= 3 ** power
        return True if n == 0 else False

#2579. Count Total Number of Colored Cells
class ColoredCellscoloredCells:
    def coloredCells(self, n):
        return 2 * (1 + n - 1) * (n - 1) + 1

#731. My Calendar II
class MyCalendarTwo:
    def __init__(self):
        self.calendar = list()

    def _findPrevMax(self, startTime, index):
        amount, max_end_time = 0, -1
        for event in self.calendar[:index]:
            if event[1] > startTime:
                amount += 1
                max_end_time = max(max_end_time, event[1])
        return amount, max_end_time
     
    def _findForwardMin(self, endTime, index):
        amount = 0
        temp_end = endTime
        for event in self.calendar[index:]:
            if event[0] >= endTime: 
                break
            elif event[0] < temp_end:
                amount += 1
            temp_end = event[1]
        return amount, self.calendar[index][0] if amount != 0 else -1
    
    def book(self, startTime, endTime):
        index = bisect.bisect(self.calendar, [startTime, endTime])
        prev_amount, prev_max_end_time = self._findPrevMax(startTime, index)
        if prev_amount > 1:
            return False
        forward_amount, forward_min_start_time = self._findForwardMin(endTime, index)
        if forward_amount > 1:
            return False
        if (forward_amount == 1
            and prev_amount == 1
            and forward_min_start_time < prev_max_end_time):
            return False
        else: 
            bisect.insort(self.calendar, [startTime, endTime])
            return True

#2965. Find Missing and Repeated Values
class FindMissingAndRepeatedValues:
    def findMissingAndRepeatedValues(self, grid):
        result = []
        checked_nums = {num for num in range(1, 1+len(grid)**2)}
        for row in grid:
            for num in row:
                if num not in checked_nums:
                    result.append(num)
                else:
                    checked_nums.remove(num)
        result.append(checked_nums.pop())
        return result

#2523. Closest Prime Numbers in Range
class ClosestPrimes:
    def closestPrimes(self, left, right):
        prime_nums = [2]
        composite_nums = set()
        for num_1 in range(3, right+1, 2):
            if num_1 not in composite_nums:
                prime_nums.append(num_1)
            for num_2 in range(num_1, 1+right//num_1, 2):
                composite_nums.add(num_1*num_2)

        result, min_diff = [-1, -1], float('inf')
        for index, num in enumerate(prime_nums[1:]):
            if (prime_nums[index] >= left) and (num <= right):
                diff = num - prime_nums[index]
                if diff < min_diff:
                    min_diff = diff
                    result = [prime_nums[index], prime_nums[index+1]]
        return result

#50. Pow(x, n)
class MyPow:
    def _pow(self, num, degree, sign):
        if degree == 0:
            return 1
        if degree == 1:
            return num ** sign
        return num ** (sign * (degree % 2)) * self._pow(num, degree//2, sign) ** 2
        
    def myPow(self, x, n):
        return self._pow(x, abs(n), 1 if n>=0 else -1)

#1922. Count Good Numbers
class CountGoodNumbers:
    def _pow(self, num, degree, modulo):
        if degree == 0:
            return 1
        if degree == 1:
            return num
        return (num ** (degree % 2) * self._pow(num, degree//2, modulo) ** 2) % modulo
        
    def countGoodNumbers(self, n):
        modulo = 10 ** 9 + 7
        return (self._pow(5, (n+1)//2, modulo) * self._pow(4, n//2, modulo)) % modulo

#2302. Count Subarrays With Score Less Than K
class CountSubarrays:
    def countSubarrays(self, nums, k):
        left, right = 0, -1
        cur_sum = 0
        result = 0
        while right < len(nums) - 1:
            right += 1
            cur_sum += nums[right]
            while cur_sum * (right - left + 1) >= k:
                cur_sum -= nums[left]
                left += 1
            result += right - left + 1
        return result

#3392. Count Subarrays of Length Three With a Condition
class CountSubarrays:
    def countSubarrays(self, nums):
        result = 0
        for index, num in enumerate(nums[2:], 2):
            result += (num == 2 * nums[index-1] + nums[index-2])
        return result

#2962. Count Subarrays Where Max Element Appears at Least K Times
class CountSubarrays:
    def countSubarrays(self, nums, k):
        max_num = max(nums)
        max_amount = 0
        left, right = 0, -1
        result = 0
        while right < len(nums) - 1:
            right += 1
            max_amount += (nums[right] == max_num)
            while max_amount == k:
                max_amount -= (nums[left] == max_num)
                result += len(nums) - right
                left += 1
        return result

#2845. Count of Interesting Subarrays
class CountInterestingSubarrays:
    def countInterestingSubarrays(self, nums, modulo, k):
        prefix = [0]
        mods = {0: 1}
        result = 0
        for num in nums:
            prefix.append((prefix[-1]+(num%modulo==k))%modulo)
            mods[prefix[-1]] = mods.setdefault(prefix[-1], 0) + 1
        for mod in prefix[:-1]:
            mods[mod] -= 1
            result += mods[(mod+k)%modulo]
        return result

#1295. Find Numbers with Even Number of Digits
class FindNumbers:
    def findNumbers(self, nums):
        return functools.reduce(lambda x, y: x+(y%2==0), 
                                map(lambda x: len(str(x)), nums),
                                0)

#838. Push Dominoes
class PushDominoes:
    def pushDominoes(self, dominoes):
        actions = {1: 'R', -1: 'L', 0: '.'}
        result = ['.'] * len(dominoes)
        active = {index: -1 if dominoes[index]=='L' else 1 
                  for index in range(len(dominoes)) if dominoes[index] in ('L', 'R')}
        while active:
            temp_active = {}
            for index, term in active.items():
                result[index] = actions[term]
                next_index = index + term
                if ((0 <= next_index < len(dominoes)) 
                    and (next_index not in active)
                    and result[next_index] == '.'):
                    temp_active[next_index] = temp_active.setdefault(next_index, 0) + term
            active = temp_active
        return ''.join(result)

#1007. Minimum Domino Rotations For Equal Row
class MinDominoRotations:
    def countReplaces(self, num, tops, bottoms):
        top_amount, bottom_amount = 0, 0
        for top, bottom in zip(tops, bottoms):
            if top != num and bottom != num:
                return float('inf')
            top_amount += (top == num)
            bottom_amount += (bottom == num)
        return min(len(tops)-top_amount, len(tops)-bottom_amount)
        
    def minDominoRotations(self, tops, bottoms):
        amount_top = self.countReplaces(tops[0], tops, bottoms)
        amount_bottom = self.countReplaces(bottoms[0], tops, bottoms)
        result = min(amount_top, amount_bottom)
        return result if result<float('inf') else -1

#1128. Number of Equivalent Domino Pairs
class NumEquivDominoPairs:
    def numEquivDominoPairs(self, dominoes):
        doms = {}
        result = 0
        for pair in dominoes:
            sorted_pair = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
            doms[sorted_pair] = doms.setdefault(sorted_pair, -1) + 1
            result += doms[sorted_pair]
        return result

#2799. Count Complete Subarrays in an Array
class CountCompleteSubarrays:
    def countCompleteSubarrays(self, nums):
        distinct_nums = set(nums)
        left, right = 0, -1
        result = 0
        subarray_nums = {}
        while right < len(nums) - 1:
            right += 1
            if nums[right] not in subarray_nums:
                distinct_nums.remove(nums[right])
                subarray_nums[nums[right]] = 1
            else:
                subarray_nums[nums[right]] += 1
            while not distinct_nums:
                subarray_nums[nums[left]] -= 1
                if subarray_nums[nums[left]] == 0:
                    subarray_nums.pop(nums[left])
                    distinct_nums.add(nums[left])
                result += len(nums) - right
                left += 1
        return result

#1399. Count Largest Group
class CountLargestGroup:
    def countLargestGroup(self, n):
        groups = {}
        max_group = 0
        for num in range(1, n+1):
            group = sum(map(lambda x: int(x), str(num)))
            groups[group] = groups.setdefault(group, 0) + 1
            max_group = max(max_group, groups[group])
        return functools.reduce(lambda x,y: x+(y==max_group), groups.values(), 0)

#790. Domino and Tromino Tiling
class NumTilings:
    def numTilings(self, n):
        status_dict = {0: [0,1,2,3], 1: [2,3], 2: [1,3], 3: [0]}
        col_status = {0: 1}
        for _ in range(n):
            next_col_status = {}
            for status, ways in col_status.items():
                for next_status in status_dict[status]:
                    next_col_status[next_status] = next_col_status.setdefault(next_status, 0) + ways
            col_status = next_col_status
        return col_status[0] % (10 ** 9 + 7)

#2145. Count the Hidden Sequences
class NumberOfArrays:
    def numberOfArrays(self, differences, lower, upper):
        del_top, del_bot = 0, 0
        bias = 0
        for diff in differences:
            bias += diff
            if bias >= 0:
                del_bot = max(del_bot, bias)
            else:
                del_top = max(del_top, -bias)
        result = upper - lower + (upper * lower <= 0) - del_top - del_bot
        return 0 if result <= 0 else result

#21. Merge Two Sorted Lists
class MergeTwoLists:
    def mergeTwoLists(self, list1, list2):
        result = 
        cur_node = result
        while list1 and list2:
            if list1.val <= list2.val:
                cur_node
                list1 = list1.next
            else:
                if cur_node is None:
                    cur_node = list2
                else:
                    cur_node.next = list2
                list2 = list2.next
            if cur_node is not None:
                cur_node = cur_node.next
        if list1:
            cur_node.next = list1
        elif list2:
            cur_node.next = list2
        return result

#2071. Maximum Number of Tasks You Can Assign
class MaxTaskAssign:
    def _canAssign(self, tasks, workers, pills, strength):
        for task in tasks[::-1]:
            if not workers:
                return False
            if workers[-1] >= task:
                workers.pop()
            else:
                target_index = bisect.bisect_left(workers, task-strength)
                if (pills > 0) and (target_index != len(workers)):
                    workers.pop(target_index)
                    pills -= 1
                else:
                    return False
        return True
        
    def maxTaskAssign(self, tasks, workers, pills, strength):
        tasks.sort()
        workers.sort()
        left, right = -1, len(tasks)
        while left < right - 1:
            middle = (left + right) // 2
            if self._canAssign(tasks[:middle+1], list(workers[-(middle+1):]), pills, strength):
                left = middle
            else:
                right = middle
        return right

#1920. Build Array from Permutation
class BuildArray:
    def buildArray(self, nums):
        return [nums[nums[index]] for index in range(len(nums))]

#2918. Minimum Equal Sum of Two Arrays After Replacing Zeros
class MinSum:
    def minSum(self, nums1, nums2):
        sum_1, zeros_1, sum_2, zeros_2 = 0, 0, 0, 0
        for index in range(max(len(nums1), len(nums2))):
            if index < len(nums1):
                sum_1 += nums1[index]
                zeros_1 += (nums1[index] == 0)
            if index < len(nums2):
                sum_2 += nums2[index]
                zeros_2 += (nums2[index] == 0)
        if ((zeros_1 == 0) and (sum_1 < sum_2 + zeros_2)) or ((zeros_2 == 0) and (sum_2 < sum_1 + zeros_1)):
            return -1
        return max(sum_1+zeros_1, sum_2+zeros_2)

#1550. Three Consecutive Odds
class ThreeConsecutiveOdds:
    def threeConsecutiveOdds(self, arr):
        index = 2
        while index < len(arr):
            if arr[index] % 2 == 0:
                index += 3
            elif arr[index-1] % 2 == 0:
                index += 2
            elif arr[index-2] % 2 == 0:
                index += 1
            else:
                return True
        return False

#3208. Alternating Groups II
class NumberOfAlternatingGroups:
    def numberOfAlternatingGroups(self, colors, k):
        colors.extend(colors[:k-1])
        duplicates = deque([])
        result = 0
        for index, num in enumerate(colors[1:k-1]):
            if num == colors[index]:
                duplicates.append(index)
        left, right = 0, k-1
        while right < len(colors):
            if colors[right] == colors[right-1]:
                duplicates.append(right-1)
            if duplicates and duplicates[0] < left:
                duplicates.popleft()
            result += not duplicates
            left += 1
            right += 1
        return result

#2379. Minimum Recolors to Get K Consecutive Black Blocks
class MinimumRecolors:
    def minimumRecolors(self, blocks, k):
        result = functools.reduce(lambda x, y: x+(y=='W'), blocks[:k], 0)
        status = result
        for left in range(1, len(blocks)-k+1):
            status -= (blocks[left-1] == 'W') - (blocks[left-1+k] == 'W')
            result = min(result, status)
        return result

#11. Container With Most Water
class MaxArea:
    def maxArea(self, height):
        left, right = 0, len(height) - 1
        result = 0
        while left < right:
            result = max(result, min(height[left], height[right])*(right-left))
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return result

#3306. Count of Substrings Containing Every Vowel and K Consonants II
class CountOfSubstrings:
    def _checkValidity(self, vowels, consonants, target):
        return (len(consonants) >= target
                and not list(filter(lambda x: not x, vowels.values())))

    def _findMinIndex(self, vowels, consonants, target):
        return min(min(vowels.values(), key=(lambda x: x[-1]))[-1], 
                   consonants[-target] if target!=0 else float('inf'))
    
    def countOfSubstrings(self, word, k):
        vowels = {'a': [], 'e': [], 'o': [], 'i': [], 'u': []}
        consonants = deque([])
        result = 0
        for index, letter in enumerate(word):
            if letter in vowels:
                vowels[letter].append(index)
            else:
                consonants.append(index)
            if self._checkValidity(vowels, consonants, k):
                min_index = self._findMinIndex(vowels, consonants, k)
                consonants.appendleft(-1)
                min_index -= consonants[-(k+1)]
                result += 0 if min_index < 0 else min_index
                consonants.popleft()
        return result

#1358. Number of Substrings Containing All Three Characters
class NumberOfSubstrings:
    def numberOfSubstrings(self, s):
        letters = {'a': -1, 'b': -1, 'c': -1}
        result = 0
        for index, letter in enumerate(s):
            letters[letter] = index
            min_index = min(letters.values())
            result += min_index + 1
        return result

#24. Swap Nodes in Pairs
class SwapPairs:
    def swapPairs(self, head):
        new_head = ListNode()
        last_node = new_head
        seq = deque([])
        while head:
            seq.appendleft(head)
            if len(seq) == 2:
                head = seq[0].next
                for _ in range(2):
                    last_node.next = seq.popleft()
                    last_node = last_node.next
            else:
                head = head.next
        if seq:
            last_node.next = seq.popleft()
            last_node = last_node.next
        last_node.next = None
        return new_head.next

#36. Valid Sudoku
class IsValidSudoku:
    def isValidSudoku(self, board):
        columns = {col: set() for col in range(9)}
        rows = {row: set() for row in range(9)}
        subblocks = {block: set() for block in range(9)}
        for row in range(9):
            for col in range(9):
                num = board[row][col]
                if num.isdigit():
                    if (num in rows[row]
                        or num in columns[col]
                        or num in subblocks[3*(row//3)+(col//3)]):
                        return False
                    rows[row].add(num)
                    columns[col].add(num)
                    subblocks[3*(row//3)+(col//3)].add(num)
        return True

#2406. Divide Intervals Into Minimum Number of Groups
class MinGroups:
    def _search(self, interval, intervals):
        left, right = 0, len(intervals) - 1
        if not intervals or interval[1] >= intervals[-1][0]:
            return right + 1
        while left < right - 1:
            middle = (left + right) // 2
            if intervals[middle][0] <= interval[1]:
                left = middle
            else:
                right = middle
        return left if intervals[left][0] > interval[1] else right
        
    def minGroups(self, intervals):
        intervals = sorted(intervals)
        result = 0
        while intervals:
            index = 0
            while index < len(intervals):
                interval = intervals.pop(index)
                index = self._search(interval, intervals)
            result += 1
        return result

#2529. Maximum Count of Positive Integer and Negative Integer
class MaximumCount:
    def maximumCount(self, nums):
        neg_amount = 0
        for num in nums:
            if num >= 0:
                break
            neg_amount += 1
        pos_amount = 0
        for num in nums[::-1]:
            if num <= 0:
                break
            pos_amount += 1
        return max(pos_amount, neg_amount)

#3356. Zero Array Transformation II
class MinZeroArray:
    def _checkArrayOnZero(self, nums, queries):
        prefix = [0] * len(nums)
        for first, last, value in queries:
            prefix[first] += value
            if last < len(nums) - 1:
                prefix[last+1] -= value
        coeff = 0
        for index in range(len(nums)):
            coeff += prefix[index]
            if nums[index] - coeff > 0:
                return False
        return True
            
    def minZeroArray(self, nums, queries):
        left, right = 0, len(queries)
        if not self._checkArrayOnZero(nums, queries):
            return -1
        while left < right - 1:
            middle = (left + right) // 2
            if self._checkArrayOnZero(nums, queries[:middle]):
                right = middle
            else:
                left = middle
        return left if self._checkArrayOnZero(nums, queries[:left]) else right

#1109. Corporate Flight Bookings
class CorpFlightBookings:
    def corpFlightBookings(self, bookings, n):
        prefix = [0] * (n + 1)
        for first, last, seats in bookings:
            prefix[first] += seats
            if last < n:
                prefix[last+1] -= seats
        result = []
        coeff = 0
        for balance in prefix[1:]:
            coeff += balance
            result.append(coeff)
        return result

#2226. Maximum Candies Allocated to K Children
class MaximumCandies:
    def _isAllocatePossible(self, candies, part, target):
        return functools.reduce(lambda x,y: x+y//part, candies, 0) >= target

    def maximumCandies(self, candies, k):
        left, right = 0, max(candies)
        while left < right - 1:
            middle = (left + right) // 2
            if self._isAllocatePossible(candies, middle, k):
                left = middle
            else:
                right = middle
        return right if self._isAllocatePossible(candies, right, k) else left

#1870. Minimum Speed to Arrive on Time
class MinSpeedOnTime:
    def _isPossible(self, speed, dist, target):
        return functools.reduce(lambda x,y: x+math.ceil(y/speed), dist[:-1], 0)+dist[-1]/speed <= target
        
    def minSpeedOnTime(self, dist, hour):
        if hour <= len(dist) - 1:
            return -1
        left, right = 0, max(max(dist), dist[-1]//(hour-len(dist)+1)+1)
        while left < right - 1:
            middle = (left + right) // 2
            if self._isPossible(middle, dist, hour):
                right = middle
            else:
                left = middle
        return right

#2594. Minimum Time to Repair Cars
class RepairCars:
    def _isPossible(self, ranks, max_time, target_cars):
        return functools.reduce(lambda x,y: x+int(math.sqrt(max_time/y)), ranks, 0) >= target_cars
        
    def repairCars(self, ranks, cars):
        left, right = 0, min(ranks) * cars ** 2
        while left < right - 1:
            middle = (left + right) // 2
            if self._isPossible(ranks, middle, cars):
                right = middle
            else:
                left = middle
        return right

#2560. House Robber IV
class MinCapability:
    def _isPossible(self, nums, target_sum, min_houses):
        checked_indexes = set()
        result = 0
        for index, num in enumerate(nums):
            if num <= target_sum and index not in checked_indexes:
                result += 1
                checked_indexes.add(index+1)
            checked_indexes.add(index)
        return result >= min_houses
        
    def minCapability(self, nums, k):
        left, right = 0, max(nums)
        while left < right - 1:
            middle = (left + right) // 2
            if self._isPossible(nums, middle, k):
                right = middle
            else:
                left = middle
        return right

#198. House Robber
class Rob:
    def rob(self, nums):
        dp = [0, 0, 0]
        dp.extend(nums)
        for index in range(3, len(dp)):
            dp[index] += max(dp[index-2], dp[index-3])
        return max(dp[-2:])

#2206. Divide Array Into Equal Pairs
class DivideArray:
    def divideArray(self, nums):
        odd_nums = set()
        for num in nums:
            if num in odd_nums:
                odd_nums.remove(num)
            else:
                odd_nums.add(num)
        return not odd_nums

#2401. Longest Nice Subarray
class LongestNiceSubarray:
    def longestNiceSubarray(self, nums):
        cur_sum = 0
        left, right = 0, 0
        result, cur_length = 1, 0
        while right < len(nums):
            if (cur_sum ^ nums[right]) == (cur_sum + nums[right]):
                cur_sum += nums[right]
                cur_length += 1
                result = max(result, cur_length)
                right += 1
            else:
                cur_sum -= nums[left]
                cur_length -= 1
                left += 1
        return result

#3191. Minimum Operations to Make Binary Array Elements Equal to One I
class MinOperations:
    def minOperations(self, nums):
        result = 0
        for index in range(len(nums)-2):
            if not nums[index]:
                nums[index], nums[index+1], nums[index+2] = True, not nums[index+1], not nums[index+2]
                result += 1
        return -1 if not nums[-1] or not nums[-2] else result

##494. Target Sum
class FindTargetSumWays:
    def findTargetSumWays(self, nums, target):
        pos_sums = {sum(nums): 1}
        for num in nums:
            for pos_sum, amount in list(pos_sums.items()):
                new_pos_sum = pos_sum - 2 * num
                pos_sums[new_pos_sum] = pos_sums.setdefault(new_pos_sum, 0) + amount
        return pos_sums.get(target, 0)

#2115. Find All Possible Recipes from Given Supplies
class FindAllRecipes:
    def findAllRecipes(self, recipes, ingredients, supplies):
        from_node, to_node = {}, {}
        for node, leafs in zip(recipes, ingredients):
            for leaf in leafs:
                from_node.setdefault(node, set()).add(leaf)
                to_node.setdefault(leaf, set()).add(node)
        leafs = {leaf for leaf, nodes in to_node.items() 
                      if leaf not in from_node}
        supplies = set(supplies)
        result = []
        while leafs:
            leaf = leafs.pop()
            if leaf in supplies:
                for node in to_node.get(leaf, []):
                    from_node[node].remove(leaf)
                    if not from_node[node]:
                        leafs.add(node)
                        result.append(node)
                        supplies.add(node)
        return result

#2140. Solving Questions With Brainpower
class MostPoints:
    def mostPoints(self, questions):
        prev_max = [0] * len(questions)
        increment = 0
        dp = [0]
        for index, question in enumerate(questions):
            points, skiped = question
            increment = max(increment, prev_max[index])
            if index + skiped + 1 < len(prev_max):
                prev_max[index+skiped+1] = max(prev_max[index+skiped+1], points+increment)
            dp.append(max(dp[-1], points+increment))
        return dp[-1]

#763. Partition Labels
class PartitionLabels:
    def partitionLabels(self, s):
        last_index = {}
        for index, letter in enumerate(s):
            last_index[letter] = index
        result = []
        begin, end = 0, 0
        for index, letter in enumerate(s):
            end = max(end, last_index[letter])
            if index == end:
                result.append(end-begin+1)
                begin = index + 1
        return result

#3169. Count Days Without Meetings
class CountDays:
    def countDays(self, days, meetings):
        meetings.append([days+1, days+1])
        meetings.sort()
        result = 0
        end = 0
        for meet_begin, meet_end in meetings:
            if end < meet_begin:
                result += meet_begin - end - 1
            end = max(end, meet_end)
        return result

#2873. Maximum Value of an Ordered Triplet I
class MaximumTripletValue:
    def maximumTripletValue(self, nums):
        max_minuend, max_diff, result = nums[0], float('-inf'), float('-inf')
        for index in range(2, len(nums)):
            max_diff = max(max_diff, max_minuend-nums[index-1])
            max_minuend = max(max_minuend, nums[index-1])
            result = max(result, max_diff*nums[index])
        return 0 if result < 0 else result

#2780. Minimum Index of a Valid Split
class MinimumIndex:
    def _findDominant(self, nums):
        num_amount = {}
        for num in nums:
            num_amount[num] = num_amount.setdefault(num, 0) + 1
            if num_amount[num] == 1 + len(nums) // 2:
                return num
        
    def minimumIndex(self, nums):
        dominant = self._findDominant(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1]+(num==dominant))
        for index in range(len(nums)):
            if ((prefix[index+1] >= 1 + (index + 1) // 2)
                and (prefix[-1] - prefix[index+1] >= 1 + (len(nums) - index - 1) // 2)):
                return index
        return -1            

#3394. Check if Grid can be Cut into Sections
class CheckValidCuts:
    def _findPossibleLines(self, n, lvl_arr):
        max_lvl_line, lines = 0, 0
        for low, top in lvl_arr:
            lines += ((low >= max_lvl_line) and (0 < max_lvl_line < n))
            max_lvl_line = max(max_lvl_line, top)
        return lines
        
    def checkValidCuts(self, n, rectangles):
        x_lvl, y_lvl = set(), set()
        for x_start, y_start, x_end, y_end in rectangles:
            x_elem, y_elem = (x_start, x_end), (y_start, y_end)
            x_lvl.add(x_elem)
            y_lvl.add(y_elem)
        x_lvl, y_lvl = sorted(x_lvl), sorted(y_lvl)
        result = False
        #Check horizontal
        hor_lines = self._findPossibleLines(n, y_lvl)
        result |= (hor_lines >= 2)
        #Check vertical    
        vert_lines = self._findPossibleLines(n, x_lvl)
        result |= (vert_lines >= 2)
        return result

#2909. Minimum Sum of Mountain Triplets II
class MinimumSum:
    def minimumSum(self, nums):
        left_min, right_min = deque([float('inf')]), deque([float('inf')])
        for num in nums[:-1]:
            left_min.append(min(left_min[-1], num))
        for num in nums[:0:-1]:
            right_min.appendleft(min(right_min[0], num))
        result = float('inf')
        for index in range(1, len(nums)-1):
            if left_min[index] < nums[index] > right_min[index]:
                result = min(result, left_min[index]+nums[index]+right_min[index])
        return -1 if result==float('inf') else result

#2551. Put Marbles in Bags
class PutMarbles:
    def putMarbles(self, weights, k):
        pairs_sum = [weights[index]+weights[index+1] for index in range(len(weights)-1)]
        pairs_sum.sort()
        return 0 if k==1 else sum(pairs_sum[-(k-1):]) - sum(pairs_sum[:(k-1)]) 

#1123. Lowest Common Ancestor of Deepest Leaves
class LcaDeepestLeaves:
    def lcaDeepestLeaves(self, root):
        queue, leafs, graph, value_node = deque([root]), set(), {}, {}
        while queue:
            leafs.clear()
            for _ in range(len(queue)):
                cur_node = queue.popleft()
                leafs.add(cur_node.val)
                value_node[cur_node.val] = cur_node
                if cur_node.left:
                    queue.append(cur_node.left)
                    graph[cur_node.left.val] = cur_node.val
                if cur_node.right:
                    queue.append(cur_node.right)
                    graph[cur_node.right.val] = cur_node.val
        while len(leafs) > 1:
            temp_leafs = set()
            while leafs:
                temp_leafs.add(graph[leafs.pop()])
            leafs = temp_leafs
        return value_node[leafs.pop()]

#2503. Maximum Number of Points From Grid Queries
class MaxPoints:
    def _getPointsToArrive(self, grid):
        points_to_arrive = [[float('inf')]*len(grid[0]) for _ in range(len(grid))]
        points_to_arrive[0][0] = grid[0][0] + 1
        heap = [(grid[0][0]+1, (0, 0))]
        while heap:
            cur_points, cur_node = heapq.heappop(heap)
            cur_row, cur_col = cur_node
            for row_diff in range(-1, 2):
                for col_diff in range(-1, 2):
                    if abs(row_diff+col_diff) == 1:
                        next_row, next_col = cur_row + row_diff, cur_col + col_diff
                        if ((0 <= next_row < len(grid))
                            and (0 <= next_col < len(grid[0]))):
                            next_points = max(grid[next_row][next_col]+1, cur_points)
                            if next_points < points_to_arrive[next_row][next_col]:
                                points_to_arrive[next_row][next_col] = next_points
                                heapq.heappush(heap, (next_points, (next_row, next_col)))
        return points_to_arrive
        
    def maxPoints(self, grid, queries):
        points_to_arrive = self._getPointsToArrive(grid)
        points_value = {}
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                points = points_to_arrive[row][col]
                points_value[points] = points_value.setdefault(points, 0) + 1
        start_points = sorted(points_value.keys())
        value = [0]
        for points in start_points:
            value.append(value[-1]+points_value[points])
        result = [value[bisect.bisect(start_points, points)] for points in queries]
        return result

#416. Partition Equal Subset Sum
class CanPartition:
    def canPartition(self, nums):
        target_sum = sum(nums)
        if target_sum % 2:
            return False
        target_sum //= 2
        pos_sums = set()
        for num in nums:
            for pos_sum in list(pos_sums):
                next_pos_sum = pos_sum + num
                pos_sums.add(next_pos_sum)
            pos_sums.add(num)
            if target_sum in pos_sums:
                return True
        return False

#698. Partition to K Equal Sum Subsets
class CanPartitionKSubsets:
    def canPartitionKSubsets(self, nums, k):
        target_sum = sum(nums)
        if target_sum % k:
            return False
        target_sum //= k
        pos_sums = []
        for index, num in enumerate(nums):
            for pos_sum, indexes in list(pos_sums):
                next_pos_sum = pos_sum + num
                next_indexes = set(indexes)
                next_indexes.add(index)
                pos_sums.append((next_pos_sum, next_indexes))
            pos_sums.append((num, {index}))
        target_indexes = [indexes for pos_sum, indexes in pos_sums if pos_sum==target_sum]
        subsets = []
        for indexes in target_indexes:
            for amount, index_set in list(subsets):
                if not index_set & indexes:
                    subsets.append((amount+1, index_set|indexes))
            subsets.append((1, indexes))
        for amount, subset in subsets:
            if amount == k:
                return True
        return False

#2685. Count the Number of Complete Components
class CountCompleteComponents:
    def countCompleteComponents(self, n, edges):
        graph = {node: set() for node in range(n)}
        for begin, end in edges:
            graph[begin].add(end)
            graph[end].add(begin)
        unchecked = set(range(n))
        result = 0
        while unchecked:
            start_node = unchecked.pop()
            graph_nodes = {start_node}
            stack = deque([start_node])
            while stack:
                cur_node = stack.pop()
                for next_node in graph[cur_node]:
                    if next_node in unchecked:
                        stack.append(next_node)
                if cur_node in unchecked:
                    unchecked.remove(cur_node)
                    graph_nodes.add(cur_node)
            result += functools.reduce(lambda x, y: x&(len(graph[y])==len(graph_nodes)-1), 
                                       graph_nodes, 
                                       True)
        return result

#3396. Minimum Number of Operations to Make Elements in Array Distinct
class MinimumOperations:
    def minimumOperations(self, nums):
        checked = set()
        result = 0
        for index, num in enumerate(nums[::-1]):
            if num in checked:
                del_amount = len(nums) - index
                result = (del_amount % 3 != 0) + del_amount // 3
                break
            checked.add(num)
        return result

#3375. Minimum Operations to Make Array Values Equal to K
class MinOperations:
    def minOperations(self, nums, k):
        nums_set, min_num = set(), float('inf')
        for num in nums:
            nums_set.add(num)
            min_num = min(min_num, num)
        return -1 if k > min_num else len(nums_set) - (k in nums_set)

#970. Powerful Integers
class PowerfulIntegers:
    def powerfulIntegers(self, x, y, bound):
        if bound <= 1:
            return []
        x_powers = [x**power for power in range(1 if x==1 else int(math.log(bound-1, x))+2)]
        y_powers = [y**power for power in range(1 if y==1 else int(math.log(bound-1, y))+2)]
        result = set()
        for x_pow in x_powers:
            for y_pow in y_powers:
                if y_pow > bound - x_pow:
                    break
                result.add(x_pow+y_pow)
        print(x_powers, y_powers)
        return list(result)

#2843. Count Symmetric Integers
class CountSymmetricIntegers:
    def countSymmetricIntegers(self, low, high):
        result = 0
        for num in range(low, high+1):
            str_num = str(num)
            if not len(str_num) % 2:
                first_part = functools.reduce(lambda x, y: x+int(y), str_num[:len(str_num)//2], 0)
                second_part = functools.reduce(lambda x, y: x+int(y), str_num[len(str_num)//2:], 0)
                result += (first_part == second_part)
        return result       

#1534. Count Good Triplets
class CountGoodTriplets:
    def _secondItems(self, item, arr, a):
        result = []
        for index, num in enumerate(arr[item[-1]+1:], item[-1]+1):
            if abs(arr[item[-1]] - num) <= a:
                result.append([item[-1], index])
        return result
        
    def _thirdItems(self, item, arr, b, c):
        result = []
        for index, num in enumerate(arr[item[-1]+1:], item[-1]+1):
            if (abs(arr[item[-1]] - num) <= b) and (abs(arr[item[-2]] - num) <= c):
                result.append([item[-2], item[-1], index])
        return result
        
    def countGoodTriplets(self, arr, a, b, c):
        result = deque([[index] for index in range(len(arr))])
        for _ in range(len(result)):
            result.extend(self._secondItems(result.popleft(), arr, a))
        for _ in range(len(result)):
            result.extend(self._thirdItems(result.popleft(), arr, b, c))
        return len(result)

#2537. Count the Number of Good Subarrays
class CountGood:        
    def countGood(self, nums, k):
        left, right = 0, 0
        total_pairs = 0
        num_amount = {}
        result = 0
        while right < len(nums):
            num_amount[nums[right]] = num_amount.setdefault(nums[right], 0) + 1
            total_pairs += num_amount[nums[right]] - 1
            while total_pairs >= k:
                num_amount[nums[left]] -= 1
                total_pairs -= num_amount[nums[left]]
                result += len(nums) - right
                left += 1
            right += 1
        return result

#38. Count and Say
class CountAndSay:
    def countAndSay(self, n):
        cur_string, cur_iter = '1', 1
        while cur_iter < n:
            new_string = [1, cur_string[0]]
            for cur_char in cur_string[1:]:
                if cur_char == new_string[-1]:
                    new_string[-2] += 1
                else:
                    new_string[-2] = str(new_string[-2])
                    new_string.extend([1, cur_char])
            new_string[-2] = str(new_string[-2])
            cur_string = ''.join(new_string)
            cur_iter += 1
        return cur_string

#2094. Finding 3-Digit Even Numbers
class FindEvenNumbers:
    def _check(self, num, digits_dict):
        digits_amount = {}
        while num > 0:
            digit = num % 10
            digits_amount[digit] = digits_amount.setdefault(digit, 0) + 1
            num //= 10
        for digit, amount in digits_amount.items():
            if digit not in digits_dict or amount > digits_dict[digit]:
                return False
        return True
        
    def findEvenNumbers(self, digits):
        digits_dict = {}
        for digit in digits:
            digits_dict[digit] = digits_dict.setdefault(digit, 0) + 1
        result = []
        for num in range(100, 1000):
            if num % 2 == 0 and self._check(num, digits_dict):
                result.append(num)
        return result

#3342. Find Minimum Time to Reach Last Room II
class MinTimeToReach:
    def minTimeToReach(self, moveTime):
        n, m = len(moveTime), len(moveTime[0])
        arrive_matrix = [[float('inf')] * m for _ in range(n)]
        arrive_matrix[0][0] = 0
        heap = [(0, 0, 0, 0)]
        while heap:
            cur_time, cur_row, cur_col, cur_step = heapq.heappop(heap)
            if (cur_row == n - 1) and (cur_col == m - 1):
                return cur_time
            for row_diff in range(-1, 2):
                for col_diff in range(-1, 2):
                    if abs(row_diff + col_diff) == 1:
                        next_row, next_col = cur_row + row_diff, cur_col + col_diff
                        if (0 <= next_row < n) and (0 <= next_col < m):
                            step_time = 1 + cur_step % 2
                            arrive_time = max(cur_time + step_time,
                                              moveTime[next_row][next_col] + step_time)
                            if arrive_time < arrive_matrix[next_row][next_col]:
                                arrive_matrix[next_row][next_col] = arrive_time
                                heapq.heappush(heap, (arrive_time, next_row, next_col, cur_step+1))

#3335. Total Characters in String After Transformations I
class LengthAfterTransformations:
    def lengthAfterTransformations(self, s, t):
        letters = {}
        for char in s:
            letter = ord(char) - ord('a')
            letters[letter] = letters.setdefault(letter, 0) + 1
        for _ in range(t):
            temp_letters = {}
            for letter, amount in letters.items():
                if amount > 0:
                    if letter == 25:
                        amount %= 10 ** 9 + 7
                        temp_letters[0] = amount
                        temp_letters[1] = temp_letters.setdefault(1, 0) + amount
                    else:
                        temp_letters[letter+1] = amount
            letters = temp_letters
        return sum(letters.values())
            

#3341. Find Minimum Time to Reach Last Room I
class MinTimeToReach:
    def minTimeToReach(self, moveTime):
        n, m = len(moveTime), len(moveTime[0])
        arrive_matrix = [[float('inf')] * m for _ in range(n)]
        arrive_matrix[0][0] = 0
        heap = [(0, 0, 0)]
        while heap:
            cur_time, cur_row, cur_col = heapq.heappop(heap)
            if (cur_row == n - 1) and (cur_col == m - 1):
                return cur_time
            for row_diff in range(-1, 2):
                for col_diff in range(-1, 2):
                    if abs(row_diff + col_diff) == 1:
                        next_row, next_col = cur_row + row_diff, cur_col + col_diff
                        if (0 <= next_row < n) and (0 <= next_col < m):
                            arrive_time = max(cur_time + 1,
                                              moveTime[next_row][next_col] + 1)
                            if arrive_time < arrive_matrix[next_row][next_col]:
                                arrive_matrix[next_row][next_col] = arrive_time
                                heapq.heappush(heap, (arrive_time, next_row, next_col))

#2900. Longest Unequal Adjacent Groups Subsequence I
class GetLongestSubsequence:
    def getLongestSubsequence(self, words, groups):
        prev_num = -1
        result = []
        for index, num in groups:
            if num != prev_num:
                result.append(words[index])
        return result

#909. Snakes And Ladders
class SnakesAndLadders:
    def _getNodeNumber(self, row, col, n):
        return n ** 2 - row * n - abs((n-1)*((n-row)%2)-col)
        
    def snakesAndLadders(self, board):
        n = len(board)
        spec_edges = {self._getNodeNumber(row, col, n): board[row][col]
                      for row in range(n) for col in range(n) if board[row][col]!=-1}
        queue = deque([1])
        visited = set()
        result = 0
        while queue:
            for _ in range(len(queue)):
                cur_node = queue.popleft()
                if cur_node == n ** 2:
                    return result
                for next_node in range(cur_node+1, min(cur_node+6, n**2)+1):
                    if next_node in spec_edges:
                        next_node = spec_edges[next_node]
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append(next_node)
            result += 1
        return -1        

#3170. Lexicographically Minimum String After Removing Stars
class ClearStars:
    def clearStars(self, s):
        result = list(s)
        min_letters = []
        letter_index = {}
        for index, letter in enumerate(s):
            if letter == '*':
                result[index] = ''
                if min_letters:
                    del_letter = heapq.heappop(min_letters)
                    del_index = letter_index[del_letter].pop()
                    result[del_index] = ''
            else:
                heapq.heappush(min_letters, letter)
                letter_index.setdefault(letter, deque([])).append(index)
        return ''.join(result)

#1857. Largest Color Value In A Directed Graph
class LargestPathValue:
    def largestPathValue(self, colors, edges):
        n = len(colors)
        graph = {} 
        leafs = set(node for node in range(n))
        node_color = {node: {colors[node]: 1} for node in range(n)}
        node_degree = {}
        result = 0
        for begin, end in edges:
            if begin == end:
                return -1
            leafs.discard(begin)
            graph.setdefault(end, set()).add(begin)
            node_degree[begin] = node_degree.setdefault(begin, 0) + 1
        while leafs:
            temp_leafs = set()
            for leaf in leafs:
                if leaf not in graph:
                    result = max(result, max(node_color[leaf].values()))
                else:
                    for parent in graph.pop(leaf):
                        for color, amount in node_color[leaf].items():
                            node_color[parent][color] = max(node_color[parent].get(color, 0),
                                                            amount+(color==colors[parent]))
                        node_degree[parent] -= 1
                        if node_degree[parent] == 0:
                            temp_leafs.add(parent)
            leafs = temp_leafs
        if graph:
            return -1
        else:
            return result

#3024. Type of Triangle
class TriangleType:
    def triangleType(self, nums):
        nums.sort()
        if nums[0] + nums[1] <= nums[2]:
            return 'none'
        elif (nums[0] == nums[1]) or (nums[1] == nums[2]):
            if nums[0] == nums[2]:
                return 'equilateral'
            return 'isosceles'
        else:
            return 'scalene'

#3355. Zero Array Transformation I
class IsZeroArray:
    def isZeroArray(self, nums, queries):
        prefix = [0] * len(nums)
        for begin, end in queries:
            prefix[begin] += 1
            if end < len(prefix) - 1:
                prefix[end+1] -= 1
        balance = 0
        for index, num in enumerate(nums):
            balance += prefix[index]
            if num > balance:
                return False
        return True

#73. Set Matrix Zeroes
class SetZeroes:
    def setZeroes(self, matrix):
        rows, cols = set(), set()
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == 0:
                    rows.add(row)
                    cols.add(col)
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if row in rows or col in cols:
                    matrix[row][col] = 0

#1857. Largest Color Value in a Directed Graph
class LargestPathValue:
    def largestPathValue(self, colors, edges):
        graph = {}
        not_checked = set(node for node in range(len(colors)))
        roots = set(not_checked)
        for begin, end in edges:
            graph.setdefault(begin, set()).add(end)
            roots.discard(end)
        result = 0
        while roots:
            stack = deque([roots.pop()])
            color_amount = {}
            while stack:
                cur_node = stack.pop()
                if cur_node not in not_checked:
                    return -1
                not_checked.remove(cur_node)
                color = colors[cur_node]
                color_amount[color] = color_amount.setdefault(color, 0) + 1
                for next_node in graph.get(cur_node, []):
                    stack.append(next_node)
            result = max(result, max(color_amount.values()))
        if not_checked:
            return -1
        return result

#135. Candy
class Candy:
    def candy(self, ratings):
        result = [0] * len(ratings)
        rating_index = {}
        for index, rating in enumerate(ratings):
            rating_index.setdefault(rating, list()).append(index)
        for rating in sorted(rating_index.keys()):
            for index in rating_index[rating]:
                left = 0 if (index == 0) or (ratings[index-1] == rating) else result[index-1]
                right = 0 if (index == len(ratings) - 1) or (ratings[index+1] == rating) else result[index+1]
                result[index] = max(left, right) + 1
        return sum(result)

#3372. Maximize The Number Of Target Nodes After Connecting Trees-I
class MaxTargetNodes:
    def _buildGraph(self, edges):
        graph = {}
        for begin, end in edges:
            graph.setdefault(begin, set()).add(end)
            graph.setdefault(end, set()).add(begin)
        return graph

    def _findNodeDegree(self, graph, node, limit):
        queue = deque([node])
        result = 0
        visited = set()
        while queue and limit >= 0:
            for _ in range(len(queue)):
                cur_node = queue.popleft()
                visited.add(cur_node)
                result += 1
                for next_node in graph[cur_node]:
                    if next_node not in visited:
                        queue.append(next_node)
            limit -= 1
        return result
            
    def maxTargetNodes(self, edges1, edges2, k):
        graph1 = self._buildGraph(edges1)
        graph2 = self._buildGraph(edges2)
        max_degree_2 = max([self._findNodeDegree(graph2, node, k-1) for node in graph2])
        result = []
        for node in graph1:
            result.append(self._findNodeDegree(graph1, node, k)+max_degree_2)
        return result

#1298. Maximum Candies You Can Get From Boxes
class MaxCandies:
    def maxCandies(self, status, candies, keys, containedBoxes, initialBoxes):
        boxes_status = {box: {'box': False, 'key': False} for box in range(len(candies))}
        for box, key in enumerate(status):
            if key == 1:
                boxes_status[box]['key'] = True
        queue = deque([])
        for box in initialBoxes:
            boxes_status[box]['box'] = True
            if boxes_status[box]['box'] and boxes_status[box]['key']:
                queue.append(box)
        result = 0
        while queue:
            cur_box = queue.popleft()
            result += candies[cur_box]
            for next_box in containedBoxes[cur_box]:
                boxes_status[next_box]['box'] = True
                if boxes_status[next_box]['box'] and boxes_status[next_box]['key']:
                    queue.append(next_box)
            for next_key in keys[cur_box]:
                if not boxes_status[next_key]['key']:
                    boxes_status[next_key]['key'] = True
                    if boxes_status[next_key]['box'] and boxes_status[next_key]['key']:
                        queue.append(next_key)
        return result

#3403. Find The Lexicographically Largest String From The Box-I
class AnswerString:
    def _findMaxString(self, string, candidate):
        index = 0
        while index < min(len(string), len(candidate)):
            if ord(string[index]) > ord(candidate[index]):
                return string
            elif ord(string[index]) < ord(candidate[index]):
                return candidate
            else:
                index += 1
        return max(string, candidate, key=len)
        
    def answerString(self, word, numFriends):
        if numFriends == 1:
            return word
        result = ''
        max_letter = max(word, key=ord)
        for index, letter in enumerate(word):
            if letter == max_letter:
                result = self._findMaxString(result, word[index:index+len(word)-numFriends+1])
        return result

#2929. Distribute Candies Among Children-II
class DistributeCandies:
    def distributeCandies(self, n, limit):
        result = 0
        for first in range(min(n, limit)+1):
            if n - first > limit * 2:
                continue
            if limit >= n - first:
                result += n - first + 1
            else:
                result += 2 * limit - n + first + 1
        return result

#1061. Lexicographically Smallest Equivalent String
class SmallestEquivalentString:
    def smallestEquivalentString(self, s1, s2, baseStr):
        queue = deque([])
        for letter_1, letter_2 in zip(s1, s2):
            set_1, set_2 = {letter_1}, {letter_2}
            for _ in range(len(queue)):
                cur_set = queue.popleft()
                if letter_1 in cur_set:
                    set_1.update(cur_set)
                elif letter_2 in cur_set:
                    set_2.update(cur_set)
                else:
                    queue.append(cur_set)
            queue.append(set_1|set_2)
        letter_min_eq = {}
        while queue:
            cur_set = queue.popleft()
            min_letter = min(cur_set)
            letter_min_eq.update((letter, min_letter) for letter in cur_set)
        return ''.join(list(letter_min_eq.get(letter, letter) for letter in baseStr))

#2434. Using A Robot To Print The Lexicographically Smallest String
class RobotWithString:
    def robotWithString(self, s):
        prefix = deque([float('inf')])
        for char in s[:0:-1]:
            prefix.appendleft(min(ord(char), prefix[0]))
        t_string, result = deque([]), []
        for index, char in enumerate(s):
            t_string.append(char)
            while t_string and (ord(t_string[-1]) <= prefix[index]):
                result.append(t_string.pop())
        return ''.join(result)

#2359. Find Closest Node To Given Two Nodes
class ClosestMeetingNode:
    def _getDistance(self, start_node, graph):
        result = {}
        length = 0
        stack = deque([start_node])
        while stack:
            cur_node = stack.pop()
            result[cur_node] = length
            next_node = graph.get(cur_node, start_node)
            if next_node not in result:
                stack.append(next_node)
            length += 1
        return result
        
    def closestMeetingNode(self, edges, node1, node2):
        graph = {}
        for begin, end in enumerate(edges):
            if end != -1:
                graph[begin] = end
        distance_1 = self._getDistance(node1, graph)
        distance_2 = self._getDistance(node2, graph)
        result = (float('inf'), -1)
        for node, distance in distance_1.items():
            if node in distance_2:
                max_dist = max(distance, distance_2[node])
                if (max_dist < result[0]) or ((max_dist == result[0]) and (node < result[1])):
                    result = (max_dist, node)                    
        return result[1]

#2894. Divisible And Non Divisible Sums Difference
class DifferenceOfSums:
    def differenceOfSums(self, n, m):
        full_sum = n * (n + 1) // 2 if n > 1 else 1
        amount = n // m
        num_2 = amount * m * (amount + 1) // 2 if amount > 0 else 0
        return full_sum - 2 * num_2

#2131. Longest Palindrome By Concatenating Two Letter Words
class LongestPalindrome:
    def longestPalindrome(self, words):
        straight, reverse = {}, {}
        for word in words:
            if word not in reverse:
                straight[word] = straight.setdefault(word, 0) + 1
                reverse.setdefault(word[::-1], 0)
            else:
                reverse[word] += 1
        result = 0
        odd_use = False
        for word in straight:
            if word == word[::-1]:
                pol_amount = straight[word] + reverse[word]
                if pol_amount % 2 == 1 and not odd_use:
                    result += 2 * pol_amount
                    odd_use = True
                else:
                    result += 2 * (pol_amount - (pol_amount % 2))
            else:
                result += 4 * min(straight[word], reverse[word[::-1]])
        return result

#440. K-th Smallest in Lexicographical Order
class FindKthNumber(object):
    def _countSteps(self, number, limit):
        min_num = int(str(number) + '0')
        max_num = int(str(number) + '9')
        result = int(number <= limit)
        while min_num <= limit:
            result += min(max_num, limit) - min_num + 1
            min_num = int(str(min_num) + '0')
            max_num = int(str(max_num) + '9')
        return result
        
    def findKthNumber(self, n, k):
        parent = 0
        bottom = 1
        while True:
            for child in range(1 if parent==0 else 10*parent , min(10*(parent+1)-1, n) + 1):
                top = bottom + self._countSteps(child, n) - 1
                if k <= top:
                    if bottom == k:
                        return child
                    parent = child
                    bottom += 1
                    break
                bottom = top + 1

#2942. Find Words Containing Character
class FindWordsContaining:
    def findWordsContaining(self, words, x):
        return [index for index, word in enumerate(words) if x in word]

#3442. Maximum Difference Between Even And Odd Frequency-I
class MaxDifference:
    def maxDifference(self, s):
        odd_freq, even_freq = {}, {}
        for letter in s:
            if letter in odd_freq:
                letter_freq = odd_freq.pop(letter)
                even_freq[letter] = letter_freq + 1
            else:
                letter_freq = even_freq.pop(letter, 0)
                odd_freq[letter] = letter_freq + 1
        return max(odd_freq.values())-min(even_freq.values())

#2138. Divide A String Into Groups Of Size K
class DivideString:
    def divideString(self, s, k, fill):
        result = []
        while s:
            result.append(s[:k])
            s = s[k:]
        result[-1] += fill * (k - len(result[-1]))
        return result

#3085. Minimum Deletions to Make String K-Special
class MinimumDeletions:
    def minimumDeletions(self, word, k):
        letter_freq = {}
        for letter in word:
            letter_freq[letter] = letter_freq.setdefault(letter, 0) + 1
        freq_arr = sorted(letter_freq.values())
        prefix = [0]
        for freq in freq_arr:
            prefix.append(prefix[-1]+freq)
        right = 0
        n = len(freq_arr)
        result = float('inf')
        for left in range(n):
            while (right < n) and (freq_arr[right] - freq_arr[left] <= k):
                right += 1
            del_right = prefix[-1] - prefix[right] - (n - right) * (freq_arr[left] + k)
            del_left = prefix[left]
            result = min(result, del_left+del_right)
            if right == n:
                return result

#2200. Find All K-Distant Indices in an Array
class FindKDistantIndices:
    def findKDistantIndices(self, nums, key, k):
        result = [-1]
        n = len(nums)
        for index, num in enumerate(nums):
            if num == key:
                left = index - k
                if left <= result[-1]:
                    left = result[-1] + 1
                right = index + k + 1
                if right > n:
                    right = n
                result.extend(range(left, right))
        result.pop(0)
        return result               

#2294. Partition Array Such That Maximum Difference Is K
class PartitionArray:
    def partitionArray(self, nums, k):
        result = 0
        index = 0
        nums.sort()
        while index < len(nums):
            index = bisect.bisect_right(nums[index:], nums[index]+k) + index
            result += 1
        return result

#2311. Longest Binary Subsequence Less Than or Equal to K
class LongestSubsequence:
    def longestSubsequence(self, s, k):
        result = 0
        for degree, char in enumerate(s[::-1]):
            k -= int(char) * (2**degree)
            result += (char=='0') or (k>=0)
        return result

#2966. Divide Array Into Arrays With Max Difference
class DivideArray:
    def divideArray(self, nums, k):
        nums.sort()
        result = []
        first_index = 0
        for num in nums[2::3]:
            if num - nums[first_index] > k:
                return []
            result.append(nums[first_index:first_index+3])
            first_index += 3
        return result

#1432. Max Difference You Can Get From Changing an Integer
class MaxDiff:
    def maxDiff(self, num):
        str_num = str(num)
        digit_to_replace = None
        for digit in str_num:
            if digit != '9':
                digit_to_replace = digit
                break
        max_num = num
        if digit_to_replace is not None:
            max_num = int(str_num.replace(digit_to_replace, '9'))
            
        digit_to_replace = None
        for digit in str_num:
            if digit not in ('1', '0'):
                digit_to_replace = digit
                break
        min_num = num
        if digit_to_replace is not None:
            min_num = int(str_num.replace(digit_to_replace, '1' if digit_to_replace==str_num[0]
                                                                else '0'))
        return max_num - min_num

#594. Longest Harmonious Subsequence
class FindLHS:
    def findLHS(self, nums):
        num_amount = {}
        for num in nums:
            num_amount[num] = num_amount.setdefault(num, 0) + 1
        result = 0
        for num, amount in num_amount.items():
            result = max(result, 
                         amount+num_amount.get(num+1, -amount),
                         amount+num_amount.get(num-1, -amount))
        return result

#3330. Find the Original Typed String I
class PossibleStringCount:
    def possibleStringCount(self, word):
        letter_amount = {}
        for letter in word:
            letter_amount[letter] = letter_amount.setdefault(letter, 0) + 1
        result = 1
        for amount in letter_amount.values():
            result += amount - 1
        return result

#3333. Find the Original Typed String-II
class PossibleStringCount:
    def _getFreqArr(self, word):
        result = [1]
        target_letter = word[0]
        for letter in word[1:]:
            if target_letter != letter:
                target_letter = letter
                result.append(0)
            result[-1] += 1
        return result

    def _getPrefixSum(self, arr):
        result = [0]
        for num in arr:
            result.append(result[-1]+num)
        return result
        
    def possibleStringCount(self, word, k):
        freq_arr = self._getFreqArr(word)
        all_comb = functools.reduce(operator.mul, freq_arr)
        modulo = 10 ** 9 + 7
        if k <= len(freq_arr):
            return all_comb % modulo
        length_amount = [0] * (k - len(freq_arr))
        length_amount[0] = 1
        for freq in freq_arr:
            prefix_arr = self._getPrefixSum(length_amount)
            length_amount = []
            for index in range(len(prefix_arr)-1):
                prefix_index = index - freq + 1
                prefix_index = 0 if prefix_index < 0 else prefix_index
                length_amount.append(prefix_arr[index+1]-prefix_arr[prefix_index])
        return (all_comb - sum(length_amount)) % modulo

#500. Keyboard Row
class FindWords:
    def _checkWord(self, word, letters):
        word = word.lower()
        for char in word:
            if char not in letters:
                return False
        return True
        
    def findWords(self, words):
        letters_set = [set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')]
        result = []
        for word in words:
            for letters in letters_set:
                if self._checkWord(word, letters):
                    result.append(word)
                    break
        return result

#3304. Find the K-th Character in String Game-I
class KthCharacter:     
    def kthCharacter(self, k):
        iter_count = 0
        while k > 1:
            n = int(math.log2(k))
            k -= 2 ** n
            iter_count += 1 if k > 0 else n
        return chr(ord('a') + iter_count % 26)

#1394. Find Lucky Integer in an Array
class FindLucky:
    def findLucky(self, arr):
        num_freq = {}
        for num in arr:
            num_freq[num] = num_freq.setdefault(num, 0) + 1
        max_lucky_num = -1
        for num, freq in num_freq.items():
            if (num == freq) and (num > max_lucky_num):
                max_lucky_num = num
        return max_lucky_num

#2008. Maximum Earnings From Taxi
class MaxTaxiEarnings:
    def maxTaxiEarnings(self, n, rides):
        rides_income = {}
        for start, end, tip in rides:
            rides_income.setdefault(start, list()).append((end, end-start+tip))
        starts_arr = deque(sorted(rides_income.keys()))
        starts_arr.append(n)
        end_prior = []
        max_income = 0
        cur_point = 0
        while starts_arr:
            for end, income in rides_income.get(cur_point, []):
                heapq.heappush(end_prior, (end, max_income+income))
            cur_point = starts_arr.popleft()
            while end_prior and (end_prior[0][0] <= cur_point):
                new_end, new_income = heapq.heappop(end_prior)
                max_income = max(max_income, new_income)
        return max_income

#1353. Maximum Number of Events That Can Be Attended
class MaxEvents:
    def _fillHeap(self, heap, values):
        for val in values:
            heapq.heappush(heap, val)
        
    def maxEvents(self, events):
        events_dict = {}
        for begin, end in events:
            events_dict.setdefault(begin, list()).append(end)
            
        begin_arr = deque(sorted(events_dict.keys()))
        begin_arr.append(float('inf'))
        cur_day = begin_arr[0]
        prior_heap = []
        result = 0
        while cur_day < float('inf'): 
            while cur_day >= begin_arr[0]:
                begin = begin_arr.popleft()
                self._fillHeap(prior_heap, events_dict[begin])
            while prior_heap and cur_day < begin_arr[0]:
                end = heapq.heappop(prior_heap)
                if end >= cur_day:
                    cur_day += 1
                    result += 1
            if not prior_heap:
                cur_day = begin_arr[0]
        return result

#1865. Finding Pairs With a Certain Sum
class FindSumPairs(object):
    def __init__(self, nums1, nums2):
        self.search_arr = list(nums2)
        self.search_dict = self._getSearchDict(nums2)
        self.nums_arr = list(nums1)

    @staticmethod
    def _getSearchDict(arr):
        result = {}
        for num in arr:
            result[num] = result.setdefault(num, 0) + 1
        return result

    def add(self, index, val):
        old_num = self.search_arr[index]
        self.search_arr[index] += val
        new_num = self.search_arr[index]
        
        self.search_dict[old_num] -= 1
        if self.search_dict[old_num] == 0:
            self.search_dict.pop(old_num)

        self.search_dict[new_num] = self.search_dict.setdefault(new_num, 0) + 1
        
    def count(self, tot):
        return sum([self.search_dict.get(tot-num, 0) for num in self.nums_arr])

#2099. Find Subsequence of Length K With the Largest Sum
class MaxSubsequence:
    def maxSubsequence(self, nums, k):
        nums_freq = {}
        for num in sorted(nums, reverse=True)[:k]:
            nums_freq[num] = nums_freq.setdefault(num, 0) + 1
        result = []
        for num in nums:
            if num in nums_freq:
                nums_freq[num] -= 1
                if nums_freq[num] == 0:
                    nums_freq.pop(num)
                result.append(num)
            if not nums_freq:
                break
        return result

#1498. Number of Subsequences That Satisfy the Given Sum Condition
class NumSubseq:
    def numSubseq(self, nums, target):
        nums.sort()
        result = 0
        for min_index, min_num in enumerate(nums):
            max_num = target - min_num
            max_index = bisect.bisect_right(nums, max_num) - 1
            if max_index < min_index:
                break
            result += 2 ** (max_index - min_index)
            result %= 10 ** 9 + 7
        return result

#3440. Reschedule Meetings for Maximum Free Time II
class MaxFreeTime:
    @staticmethod
    def _getFreePlaces(prefix):
        result = [0]
        for free_place in prefix[:-3:2]:
            result.append(max(result[-1], free_place))
        return result            
    
    def maxFreeTime(self, eventTime, startTime, endTime):
        cur_time = 0
        prefix = []
        for start, end in zip(startTime, endTime):
            prefix.extend([start-cur_time, start-end])
            cur_time = end
        prefix.append(eventTime-cur_time)
        left_to_right, right_to_left = self._getFreePlaces(prefix), self._getFreePlaces(prefix[::-1])[::-1]
        free_places = [max(left, right) for left, right in zip(left_to_right, right_to_left)]
        result = 0
        for index, meeting in enumerate(prefix[1::2]):
            left = index * 2
            right = left + 2
            free_space = prefix[left] + prefix[right] - meeting
            if free_places[index] + meeting < 0:
                free_space += meeting
            result = max(result, free_space)
        return result

#3136. Valid Word
class IsValid:
    def isValid(self, word):
        word = word.lower()
        if len(word) < 3:
            return False
        match = re.search(r'[^0-9a-z]', word)
        if match is not None:
            return False
        match = re.search(r'[aeiou]', word)
        if match is None:
            return False
        match = re.search(r'[^aeiou]', word)
        if match is None:
            return False
        return True

#3202. Find the Maximum Length of Valid Subsequence II
class MaximumLength:
    def maximumLength(self, nums, k):
        print(len(nums))
        mods = [num % k for num in nums]
        checked = set()
        result = 0
        for index, mod in enumerate(mods):
            if mod not in checked:
                combinations = deque([(None, mod, index, 1)])
                while combinations:
                    prev_mod, cur_mod, cur_index, cur_length = combinations.pop()
                    result = max(result, cur_length)
                    cur_checked = set()
                    for next_index, next_mod in enumerate(mods[cur_index+1:], cur_index+1):
                        if (next_mod not in cur_checked) and ((prev_mod is None) 
                                                              or (prev_mod == next_mod)):
                            combinations.append((cur_mod, next_mod, next_index, cur_length+1))
                            cur_checked.add(next_mod)
            checked.add(mod)
        return result

#1751. Maximum Number of Events That Can Be Attended II
class MaxValue:
    def maxValue(self, events, k):
        start_end = {}
        for start, end, val in events:
            start_end.setdefault(start, list()).append((end, val))
        start_days = deque(sorted(start_end.keys()))
        start_days.append(float('inf'))
        best_schedule = {amount: (0, []) for amount in range(1, k+1)} #(val, visited_events)
        end_days_prior = []
        while start_days:
            cur_day = start_days.popleft()
            while end_days_prior and (end_days_prior[0][0] < cur_day):
                end, val, visited_events = heapq.heappop(end_days_prior)
                amount = len(visited_events)
                best_schedule[amount] = max(best_schedule[amount],
                                            (val, visited_events),
                                            key=(lambda x: x[0]))
            for cur_end, cur_val in start_end.get(cur_day, []):
                for best_val, best_visited_events in best_schedule.values():
                    new_visited_events = list(best_visited_events)
                    replace_event = 0
                    if len(new_visited_events) == k:
                        replace_event = heapq.heappop(new_visited_events)
                    heapq.heappush(new_visited_events, cur_val)
                    new_val = best_val + cur_val - replace_event
                    heapq.heappush(end_days_prior, (cur_end, new_val, new_visited_events))
        return max(best_schedule.values())[0]

#3439. Reschedule Meetings for Maximum Free Time I
class MaxFreeTime:
    @staticmethod
    def _getPrefix(startTime, endTime, eventTime):
        result = []
        cur_time = 0
        for start, end in zip(startTime, endTime):
            result.append(start-cur_time)
            cur_time = end
        result.append(eventTime-cur_time)
        return result
        
    def maxFreeTime(self, eventTime, k, startTime, endTime):
        prefix = self._getPrefix(startTime, endTime, eventTime)
        left, right = 0, k
        free_space = sum(prefix[left:right+1])
        result = free_space
        while right < len(prefix) - 1:
            right += 1
            free_space += (-prefix[left] + prefix[right])
            left += 1
            result = max(result, free_space)
        return result

#2402. Meeting Rooms III
class MostBooked:
    @staticmethod
    def _makeMeeting(active, waiting, rooms, penalty, start_end):
        start = heapq.heappop(waiting)
        end = penalty + start_end[start]
        room = heapq.heappop(rooms)
        heapq.heappush(active, (end, room))

    @staticmethod
    def _releaseRoom(active, rooms, result):
        end, room = heapq.heappop(active)
        heapq.heappush(rooms, room)
        result[room] += 1
    
    def mostBooked(self, n, meetings):
        start_end = {start: end for start, end in meetings}
        rooms = list(range(n))
        heapq.heapify(rooms)
        waiting = list(start_end.keys())
        heapq.heapify(waiting)
        active = []
        result = {room: 0 for room in range(n)}
        while waiting or active:
            #Set Cur Time
            cur_time = waiting[0] if waiting else active[0][0]
            #Release Rooms
            while active and (active[0][0] <= cur_time):
                self._releaseRoom(active, rooms, result)
            if waiting:
                #Attend Room
                penalty = 0
                if not rooms:
                    penalty = active[0][0] - waiting[0]
                    self._releaseRoom(active, rooms, result)
                self._makeMeeting(active, waiting, rooms, penalty, start_end)
        return min(filter(lambda x: x[1]==max(result.values()), result.items()))[0]

#3201. Find the Maximum Length of Valid Subsequence I
class MaximumLength:
    def maximumLength(self, nums):
        even, odd, variable = 0, 0, 1
        last_var = nums[0] % 2
        for num in nums:
            even += (num % 2 == 0)
            odd += (num % 2 == 1)
            variable += (num % 2 != last_var)
            last_var = (last_var + (num % 2 != last_var)) % 2
        return max(even, odd, variable)

#2410. Maximum Matching of Players With Trainers
class MatchPlayersAndTrainers:
    def matchPlayersAndTrainers(self, players, trainers):
        players = deque(sorted(players))
        trainers = deque(sorted(trainers))
        result = 0
        while players and trainers:
            trainer = trainers.popleft()
            if trainer >= players[0]:
                players.popleft()
                result += 1
        return result

#1957. Delete Characters to Make Fancy String
class MakeFancyString:
    def makeFancyString(self, s):
        result = list(s[:2])
        for index, char in enumerate(s[2:], 2):
            if not ((char == s[index-1]) and (char == s[index-2])):
                result.append(char)
        return ''.join(result)

#1695. Maximum Erasure Value
class MaximumUniqueSubarray:
    def maximumUniqueSubarray(self, nums):
        left, right = 0, 0
        result = 0
        cur_subarray, cur_sum = {}, 0
        while right < len(nums):
            cur_subarray[nums[right]] = cur_subarray.setdefault(nums[right], 0) + 1
            cur_sum += nums[right]
            while cur_subarray[nums[right]] > 1:
                cur_sum -= nums[left]
                cur_subarray[nums[left]] -= 1
                left += 1
            result = max(result, cur_sum)
            right += 1
        return result

#1717. Maximum Score From Removing Substrings
class MaximumGain:
    def _checkLastTwo(self, substring, target):
        if len(substring) < 2:
            return False
        return (substring[-2] + substring[-1]) == target

    def _removeSubstring(self, string, target, score):
        result = 0
        substring = deque([])
        for letter in string:
            substring.append(letter)
            while self._checkLastTwo(substring, target):
                substring.pop()
                substring.pop()
                result += score
        return result, ''.join(substring)
        
    def maximumGain(self, s, x, y):
        gain = {'ab': x, 'ba': y}
        result = 0
        for target, score in sorted(gain.items(), key=(lambda item: item[1]), reverse=True):
            res, s = self._removeSubstring(s, target, score)
            result += res
        return result

#2322. Minimum Score After Removals on a Tree
class MinimumScore:
    def _initGraphLeafs(self, edges):
        graph, leafs = dict(), set(range(len(edges)+1))
        for begin, end in edges:
            graph.setdefault(begin, set()).add(end)
            graph.setdefault(end, set()).add(begin)
            if len(graph[begin]) > 1:
                leafs.discard(begin)
            if len(graph[end]) > 1:
                leafs.discard(end)
        return graph, leafs
        
    def _checkAllRemovals(self, graph, remain_leafs, init_nums, full_score, first_score):
        node_degree = {node: len(neighbours) for node, neighbours in graph.items()}
        leafs = set(remain_leafs)
        nums = list(init_nums)
        result = float('inf')
        while leafs:
            second_comp = leafs.pop()
            second_score = nums[second_comp]
            third_score = full_score ^ first_score ^ second_score
            score = max(first_score, second_score, third_score) - min(first_score, second_score, third_score)
            result = min(result, score)
            node_degree[second_comp] -= 1
            second_neighbour = list(filter(lambda node: node_degree[node]>0, graph[second_comp]))[0]
            node_degree[second_neighbour] -= 1
            if node_degree[second_neighbour] == 0:
                leafs.remove(second_neighbour)
            elif node_degree[second_neighbour] == 1:
                leafs.add(second_neighbour)
            nums[second_neighbour] ^= nums[second_comp]     
        return result
        
    def minimumScore(self, nums, edges):
        graph, leafs = self._initGraphLeafs(edges)
        full_score = functools.reduce(lambda x,y: x^y, nums)
        result = float('inf')
        while leafs:
            first_comp = leafs.pop()
            first_score = nums[first_comp]
            first_neighbour = graph.pop(first_comp).pop()
            graph[first_neighbour].remove(first_comp)
            if not graph[first_neighbour]:
                graph.pop(first_neighbour)
                leafs.remove(first_neighbour)
            if len(graph.get(first_neighbour, [])) == 1:
                leafs.add(first_neighbour)
            score = self._checkAllRemovals(graph, leafs, nums, full_score, first_score)
            result = min(result, score)
            nums[first_neighbour] ^= nums[first_comp]
        return result

#3487. Maximum Unique Subarray Sum After Deletion
class MaxSum:
    def maxSum(self, nums):
        subarray = set()
        max_elem = float('-inf')
        result = 0
        for num in nums:
            max_elem = max(num, max_elem)
            if (num > 0) and (num not in subarray):
                result += num
                subarray.add(num)
        return result if subarray else max_elem

#2411. Smallest Subarrays With Maximum Bitwise OR
class SmallestSubarrays:
    @staticmethod
    def _binRepr(num):
        return {index: 1 for index, bin_num in enumerate(bin(num)[:1:-1]) if bin_num == '1'}

    @staticmethod
    def _binCompare(subseq, target):
        for index in target.keys():
            if index not in subseq:
                return False
        return True
            
    def smallestSubarrays(self, nums):
        prefix = deque([nums[-1]])
        for num in nums[-2::-1]:
            prefix.appendleft(max(prefix[0], prefix[0]|num))
        prefix = list(map(self._binRepr, prefix))
        bin_nums = list(map(self._binRepr, nums))
        left, right = 0, 0
        result = []
        cur_subseq = dict(bin_nums[0])
        while left < len(bin_nums):
            while (right < left) or not self._binCompare(cur_subseq, prefix[left]):
                right += 1
                for add_index in bin_nums[right].keys():
                    cur_subseq[add_index] = cur_subseq.setdefault(add_index, 0) + 1
            result.append(right-left+1)
            for del_index in bin_nums[left].keys():
                cur_subseq[del_index] -= 1
                if cur_subseq[del_index] == 0:
                    cur_subseq.pop(del_index)
            left += 1
        return result

