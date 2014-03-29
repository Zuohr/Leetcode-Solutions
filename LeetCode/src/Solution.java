import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

import org.junit.Test;

public class Solution {

	/**
	 * $(Single Number)
	 * 
	 * Given an array of integers, every element appears twice except for one.
	 * Find that single one.
	 * 
	 * Note: Your algorithm should have a linear runtime complexity. Could you
	 * implement it without using extra memory?
	 */

	public int singleNumber(int[] A) {
		int ret = 0;
		for (int i = 0; i < A.length; i++) {
			ret ^= A[i];
		}
		return ret;
	}

	/**
	 * $(Maximum Depth of Binary Tree)
	 * 
	 * Given a binary tree, find its maximum depth.
	 * 
	 * The maximum depth is the number of nodes along the longest path from the
	 * root node down to the farthest leaf node.
	 */

	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
	}

	/**
	 * $(Same Tree)
	 * 
	 * Given two binary trees, write a function to check if they are equal or
	 * not.
	 * 
	 * Two binary trees are considered equal if they are structurally identical
	 * and the nodes have the same value.
	 */

	class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}

	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null) {
			return q == null;
		} else if (q == null) {
			return p == null;
		}

		if (p.val == q.val) {
			if (isSameTree(p.left, q.left)) {
				return isSameTree(p.right, q.right);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

	/**
	 * $(Reverse Integer)
	 * 
	 * Reverse digits of an integer. Example1: x = 123, return 321 Example2: x =
	 * -123, return -321
	 * 
	 * Have you thought about this? Here are some good questions to ask before
	 * coding. Bonus points for you if you have already thought through this!
	 * 
	 * If the integer's last digit is 0, what should the output be? ie, cases
	 * such as 10, 100.
	 * 
	 * Did you notice that the reversed integer might overflow? Assume the input
	 * is a 32-bit integer, then the reverse of 1000000003 overflows. How should
	 * you handle such cases?
	 * 
	 * Throw an exception? Good, but what if throwing an exception is not an
	 * option? You would then have to re-design the function (ie, add an extra
	 * parameter).
	 */

	public int reverse(int x) {
		boolean negative = false;
		if (x < 0) {
			negative = true;
			x = ~x + 1;
		}

		String str = String.valueOf(x);
		String max = String.valueOf(Integer.MAX_VALUE);
		int len = str.length();
		char[] chs = str.toCharArray();
		for (int i = 0; i < len / 2; i++) {
			char temp = chs[i];
			chs[i] = chs[len - 1 - i];
			chs[len - 1 - i] = temp;
		}
		str = new String(chs);

		if (str.length() == max.length() && str.compareTo(max) > 0) {
			if (negative) {
				return Integer.MIN_VALUE;
			} else {
				return Integer.MAX_VALUE;
			}
		}

		int num = Integer.parseInt(str);
		if (negative) {
			num = ~num + 1;
		}
		return num;
	}

	/**
	 * $(Best Time to Buy and Sell Stock II)
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i. Design an algorithm to find the maximum profit. You may
	 * complete as many transactions as you like (ie, buy one and sell one share
	 * of the stock multiple times). However, you may not engage in multiple
	 * transactions at the same time (ie, you must sell the stock before you buy
	 * again).
	 */

	public int maxProfit2(int[] prices) {
		int index = 0;
		int profit = 0;
		while (index < prices.length - 1) {
			int buy, sell;
			while (index < prices.length - 1
					&& prices[index] >= prices[index + 1]) {
				index++;
			}
			buy = prices[index];
			while (index < prices.length - 1
					&& prices[index] < prices[index + 1]) {
				index++;
			}
			sell = prices[index];
			if (sell - buy > 0) {
				profit += sell - buy;
			}
		}
		return profit;
	}

	/**
	 * $(Search Insert Position)
	 * 
	 * Given a sorted array and a target value, return the index if the target
	 * is found. If not, return the index where it would be if it were inserted
	 * in order.
	 * 
	 * You may assume no duplicates in the array.
	 * 
	 * Here are few examples. [1,3,5,6], 5 → 2 [1,3,5,6], 2 → 1 [1,3,5,6], 7 → 4
	 * [1,3,5,6], 0 → 0
	 */
	public int searchInsert(int[] A, int target) {
		int start = 0, end = A.length - 1;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			if (target == A[mid]) {
				return mid;
			} else if (target < A[mid]) {
				end = mid - 1;
			} else {
				start = mid + 1;
			}
		}
		return start;
	}

	class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
			next = null;
		}
	}

	/**
	 * $(Remove Duplicates from Sorted List)
	 * 
	 * Given a sorted linked list, delete all duplicates such that each element
	 * appear only once.
	 * 
	 * For example, Given 1->1->2, return 1->2. Given 1->1->2->3->3, return
	 * 1->2->3.
	 */

	public ListNode deleteDuplicatesI(ListNode head) {
		ListNode ptr = head;
		while (ptr != null) {
			if (ptr.next != null && ptr.next.val == ptr.val) {
				ptr.next = ptr.next.next;
			} else {
				ptr = ptr.next;
			}
		}
		return head;
	}

	/**
	 * $(Remove Duplicates from Sorted List II)
	 * 
	 * Given a sorted linked list, delete all nodes that have duplicate numbers,
	 * leaving only distinct numbers from the original list.
	 * 
	 * For example, Given 1->2->3->3->4->4->5, return 1->2->5. Given
	 * 1->1->1->2->3, return 2->3.
	 */

	public ListNode deleteDuplicates(ListNode head) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode ptr = dummy;

		while (ptr != null && ptr.next != null && ptr.next.next != null) {
			if (ptr.next.val == ptr.next.next.val) {
				int val = ptr.next.val;
				while (ptr.next != null && ptr.next.val == val) {
					ptr.next = ptr.next.next;
				}
			} else {
				ptr = ptr.next;
			}
		}

		return dummy.next;
	}

	/**
	 * $(Unique Binary Search Trees)
	 * 
	 * Given n, how many structurally unique BST's (binary search trees) that
	 * store values 1...n? (remark : unique shape, position of number does not
	 * matter)
	 * 
	 * For example, Given n = 3, there are a total of 5 unique BST's.
	 */

	public int numTrees(int n) {
		int[] dp = new int[n + 1];
		dp[0] = 1;
		for (int i = 1; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				dp[i] += dp[j] * dp[i - 1 - j];
			}
		}

		return dp[n];
	}

	/**
	 * $(Populating Next Right Pointers in Each Node)
	 * 
	 * Given a binary tree
	 * 
	 * Populate each next pointer to point to its next right node(right
	 * sibling). If there is no next right node, the next pointer should be set
	 * to NULL.
	 * 
	 * Initially, all next pointers are set to NULL.
	 * 
	 * Note:
	 * 
	 * You may only use constant extra space. You may assume that it is a
	 * perfect binary tree (ie, all leaves are at the same level, and every
	 * parent has two children).
	 */

	class TreeLinkNode {
		int val;
		TreeLinkNode left, right, next;

		TreeLinkNode(int x) {
			val = x;
		}
	}

	public void connect(TreeLinkNode root) {
		if (root == null) {
			return;
		}
		TreeLinkNode pvt = root;
		while (pvt.left != null) {
			TreeLinkNode ptr = pvt;
			while (ptr != null) {
				ptr.left.next = ptr.right;
				if (ptr.next != null) {
					ptr.right.next = ptr.next.left;
				}
				ptr = ptr.next;
			}
			pvt = pvt.left;
		}
	}

	/**
	 * $(Symmetric Tree)
	 * 
	 * Given a binary tree, check whether it is a mirror of itself (ie,
	 * symmetric around its center). Bonus points if you could solve it both
	 * recursively and iteratively.
	 */

	public boolean isSymmetric(TreeNode root) {
		return root == null || isSymmetric(root.left, root.right);
	}

	public boolean isSymmetric(TreeNode left, TreeNode right) {
		if (left == null) {
			return right == null;
		} else if (right == null) {
			return left == null;
		}
		if (left.val == right.val && isSymmetric(left.left, right.right)) {
			return isSymmetric(left.right, right.left);
		} else {
			return false;
		}
	}

	/**
	 * $(Climbing Stairs)
	 * 
	 * You are climbing a stair case. It takes n steps to reach to the top.
	 * 
	 * Each time you can either climb 1 or 2 steps. In how many distinct ways
	 * can you climb to the top?
	 */

	public int climbStairs(int n) {
		int[] m = new int[n + 1];
		m[0] = 1;
		m[1] = 1;
		for (int i = 2; i <= n; i++) {
			m[i] = m[i - 1] + m[i - 2];
		}
		return m[n];
	}

	/**
	 * $(Merge Two Sorted Lists)
	 * 
	 * Merge two sorted linked lists and return it as a new list. The new list
	 * should be made by splicing together the nodes of the first two lists.
	 */

	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		ListNode dummy = new ListNode(0), ptr = dummy;
		while (l1 != null && l2 != null) {
			if (l1.val > l2.val) {
				ptr.next = l2;
				ptr = l2;
				l2 = l2.next;
			} else {
				ptr.next = l1;
				ptr = l1;
				l1 = l1.next;
			}
		}
		if (l1 == null) {
			ptr.next = l2;
		} else {
			ptr.next = l1;
		}
		return dummy.next;
	}

	/**
	 * $(Remove Duplicates from Sorted Array)
	 * 
	 * Given a sorted array, remove the duplicates in place such that each
	 * element appear only once and return the new length.
	 * 
	 * Do not allocate extra space for another array, you must do this in place
	 * with constant memory.
	 * 
	 * For example, Given input array A = [1,1,2],
	 * 
	 * Your function should return length = 2, and A is now [1,2].
	 */

	public int removeDuplicates(int[] A) {
		if (A == null || A.length == 0) {
			return 0;
		}
		int read = 1, write = 0, len = A.length;
		while (read < len) {
			if (A[read] != A[write]) {
				A[++write] = A[read];
			}
			read++;
		}
		return write + 1;
	}

	/**
	 * $(Remove Element)
	 * 
	 * Given an array and a value, remove all instances of that value in place
	 * and return the new length.
	 * 
	 * The order of elements can be changed. It doesn't matter what you leave
	 * beyond the new length.
	 */

	public int removeElement(int[] A, int elem) {
		int write = 0, read = 0, len = A.length;
		while (read < len) {
			if (A[read] != elem) {
				A[write++] = A[read];
			}
			read++;
		}
		return write;
	}

	/**
	 * $(Maximum Subarray)
	 * 
	 * Find the contiguous subarray within an array (containing at least one
	 * number) which has the largest sum.
	 * 
	 * For example, given the array [ -2, 1, -3, 4, -1, 2, 1, -5, 4], the
	 * contiguous subarray [4,-1,2,1] has the largest sum = 6.
	 * 
	 * Remark: there is a O(n) solution.
	 */

	public int maxSubArray(int[] A) {
		if (A == null || A.length == 0) {
			return 0;
		}
		if (A.length == 1) {
			return A[0];
		}
		return maxSubArray(A, 0, A.length - 1);

	}

	private int maxSubArray(int[] A, int start, int end) {
		if (start == end) {
			return A[start];
		}
		int mid = (start + end) / 2;
		int leftMax = maxSubArray(A, start, mid);
		int rightMax = maxSubArray(A, mid + 1, end);
		int midMax = mergeMax(A, start, mid, end);

		return Math.max(midMax, Math.max(leftMax, rightMax));
	}

	private int mergeMax(int[] A, int start, int mid, int end) {
		int lmax = A[mid];
		int l = mid - 1;
		int sum = lmax;
		while (l >= start) {
			sum += A[l--];
			if (sum > lmax) {
				lmax = sum;
			}
		}

		int rmax = A[mid + 1];
		sum = rmax;
		int r = mid + 2;
		while (r <= end) {
			sum += A[r++];
			if (sum > rmax) {
				rmax = sum;
			}
		}

		return lmax + rmax;
	}

	public int maxSubArrayLinear(int[] A) {
		int max = Integer.MIN_VALUE, sofar = 0;
		for (int n : A) {
			sofar += n;
			max = Math.max(max, sofar);
			if (sofar < 0) {
				sofar = 0;
			}
		}
		return max;
	}

	/**
	 * $(Roman to Integer)
	 * 
	 * Given a roman numeral, convert it to an integer.
	 * 
	 * Input is guaranteed to be within the range from 1 to 3999.
	 */

	public int romanToInt(String s) {
		if (s == null || s.isEmpty()) {
			return 0;
		}

		HashMap<Character, Integer> m = new HashMap<Character, Integer>();
		m.put('M', 1000);
		m.put('D', 500);
		m.put('C', 100);
		m.put('L', 50);
		m.put('X', 10);
		m.put('V', 5);
		m.put('I', 1);

		char[] chs = s.toCharArray();
		int sum = 0, len = chs.length;
		for (int i = 0; i < len;) {
			if (i < len - 1 && m.get(chs[i]) < m.get(chs[i + 1])) {
				sum += m.get(chs[i + 1]) - m.get(chs[i]);
				i += 2;
			} else {
				sum += m.get(chs[i]);
				i++;
			}
		}

		return sum;
	}

	/**
	 * $(Merge Sorted Array)
	 * 
	 * Given two sorted integer arrays A and B, merge B into A as one sorted
	 * array.
	 * 
	 * Note: You may assume that A has enough space to hold additional elements
	 * from B. The number of elements initialized in A and B are m and n
	 * respectively.
	 */

	public void merge(int A[], int m, int B[], int n) {
		int index = A.length - 1, pa = m - 1, pb = n - 1;
		while (pa >= 0 && pb >= 0) {
			if (A[pa] > B[pb]) {
				A[index--] = A[pa];
				pa--;
			} else {
				A[index--] = B[pb];
				pb--;
			}
		}
		if (pa < 0) {
			while (pb >= 0) {
				A[index--] = B[pb--];
			}
		}
	}

	/**
	 * $(Surrounded Region)
	 * 
	 * Given a 2D board containing 'X' and 'O', capture all regions surrounded
	 * by 'X'.
	 * 
	 * A region is captured by flipping all 'O's into 'X's in that surrounded
	 * region.
	 */

	public void solve(char[][] board) {
		if (board == null || board.length == 0 || board[0] == null
				|| board[0].length == 0) {
			return;
		}

		int ht = board.length, wd = board[0].length;
		for (int col = 0; col < wd; col++) {
			if (board[0][col] == 'O') {
				flipBFS(board, 0, col);
			}
			if (board[ht - 1][col] == 'O') {
				flipBFS(board, ht - 1, col);
			}
		}
		for (int row = 0; row < ht; row++) {
			if (board[row][0] == 'O') {
				flipBFS(board, row, 0);
			}
			if (board[row][wd - 1] == 'O') {
				flipBFS(board, row, wd - 1);
			}
		}

		for (int row = 0; row < ht; row++) {
			for (int col = 0; col < wd; col++) {
				if (board[row][col] == 'N') {
					board[row][col] = 'O';
				} else if (board[row][col] == 'O') {
					board[row][col] = 'X';
				}
			}
		}
	}

	private void flipBFS(char[][] board, int row, int col) {
		/*
		 * a 'vst' set is not required since the change of pixel can function as
		 * a mark
		 */
		Queue<Integer> q = new LinkedList<Integer>();
		int ht = board.length, wd = board[0].length;
		q.add(row * wd + col);

		while (!q.isEmpty()) {
			int curr = q.remove();
			int currRow = curr / wd, currCol = curr % wd;
			if (board[currRow][currCol] != 'O') {
				continue;
			}
			board[currRow][currCol] = 'N';

			int[][] dirs = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
			for (int i = 0; i < 4; i++) {
				int nextRow = currRow + dirs[i][0], nextCol = currCol
						+ dirs[i][1];
				int nextNum = nextRow * wd + nextCol;
				if (nextRow >= 0 && nextRow < ht && nextCol >= 0
						&& nextCol < wd && board[nextRow][nextCol] == 'O') {
					q.add(nextNum);
				}
			}
		}
	}

	/**
	 * $(Gray Code)
	 * 
	 * The gray code is a binary numeral system where two successive values
	 * differ in only one bit.
	 * 
	 * Given a non-negative integer n representing the total number of bits in
	 * the code, print the sequence of gray code. A gray code sequence must
	 * begin with 0.
	 * 
	 * For example, given n = 2, return [0,1,3,2]. Its gray code sequence is: 00
	 * - 0 01 - 1 11 - 3 10 - 2 Note: For a given n, a gray code sequence is not
	 * uniquely defined.
	 * 
	 * For example, [0,2,3,1] is also a valid gray code sequence according to
	 * the above definition.
	 * 
	 * For now, the judge is able to judge based on one instance of gray code
	 * sequence. Sorry about that.
	 */

	public ArrayList<Integer> grayCodeQuadratic(int n) {
		int size = (int) Math.pow(2, (double) n);
		ArrayList<Integer> nums = new ArrayList<Integer>(size);
		if (n == 0) {
			nums.add(0);
		} else {
			for (int i = 0; i < size; i++) {
				nums.add(0);
			}
			nums.set(0, 0);
			nums.set(1, 1);
			for (int bits = 2, len = 2; bits <= n; bits++, len *= 2) {
				int ext = len * 2 - 1;
				for (int i = 0; i < len; i++) {
					nums.set(ext - i, nums.get(i));
				}
				int mask = 1 << bits - 1;
				for (int i = 0; i < len; i++) {
					nums.set(len + i, nums.get(len + i) | mask);
				}
			}
		}
		return nums;
	}

	public ArrayList<Integer> grayCode(int n) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (n >= 0) {
			int num = 1 << n;
			for (int i = 0; i < num; i++) {
				result.add(i >> 1 ^ i);
			}
		}
		return result;
	}

	/**
	 * Write an efficient algorithm that searches for a value in an m x n
	 * matrix. This matrix has the following properties:
	 * 
	 * Integers in each row are sorted from left to right. The first integer of
	 * each row is greater than the last integer of the previous row.
	 */

	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null
				|| matrix[0].length == 0) {
			return false;
		}

		int width = matrix[0].length, len = matrix.length * width;
		int start = 0, end = len - 1;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			int midVal = matrix[mid / width][mid % width];
			if (midVal == target) {
				return true;
			} else if (midVal > target) {
				end = mid - 1;
			} else {
				start = mid + 1;
			}
		}

		return false;
	}

	/**
	 * $(Permutations)
	 * 
	 * Given a collection of numbers, return all possible permutations.
	 */

	/*
	 * DFS
	 */
	public ArrayList<ArrayList<Integer>> permuteDFS(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null) {
			return result;
		}

		Arrays.sort(num);
		ArrayList<Integer> permutation = new ArrayList<Integer>();
		permuteDFS(result, permutation, num);

		return result;
	}

	private void permuteDFS(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> permutation, int[] num) {
		if (permutation.size() == num.length) {
			result.add(new ArrayList<Integer>(permutation));
			return;
		}

		for (int i = 0; i < num.length; i++) {
			if (!permutation.contains(num[i])) {
				permutation.add(num[i]);
				permuteDFS(result, permutation, num);
				permutation.remove(permutation.size() - 1);
			}
		}
	}

	/*
	 * (optimal) use next_permutation algorithm
	 */
	public ArrayList<ArrayList<Integer>> permute(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null) {
			return result;
		}

		Arrays.sort(num);
		do {
			ArrayList<Integer> permutation = new ArrayList<Integer>();
			for (int n : num) {
				permutation.add(n);
			}
			result.add(permutation);
		} while (nextPermutation(num, 0, num.length - 1));

		return result;
	}

	private boolean nextPermutation(int[] num, int start, int end) {
		int tail = end;
		while (tail > start && num[tail] <= num[tail - 1]) {
			tail--;
		}
		if (tail <= start) {
			return false;
		}
		int head = tail - 1, ptr = end;
		while (ptr > head && num[ptr] <= num[head]) {
			ptr--;
		}
		arraySwap(num, head, ptr);
		arrayReverse(num, tail, end);
		return true;
	}

	public void arraySwap(int[] num, int p1, int p2) {
		if (p1 == p2) {
			return;
		}
		num[p1] ^= num[p2];
		num[p2] ^= num[p1];
		num[p1] ^= num[p2];
	}

	public void arrayReverse(int[] num, int start, int end) {
		if (start == end) {
			return;
		}

		int len = (end - start + 1) / 2;
		for (int i = 0; i < len; i++) {
			arraySwap(num, start + i, end - i);
		}
	}

	/**
	 * $(Permutation Sequence)
	 * 
	 * The set [1,2,3,…,n] contains a total of n! unique permutations.
	 * 
	 * By listing and labeling all of the permutations in order, We get the
	 * following sequence (ie, for n = 3):
	 * 
	 * "123" "132" "213" "231" "312" "321"
	 * 
	 * Given n and k, return the kth permutation sequence.
	 * 
	 * Note: Given n will be between 1 and 9 inclusive.
	 */

	public String getPermutation(int n, int k) {
		int totalNum = 1;
		for (int i = 1; i <= n; i++) {
			totalNum *= i;
		}
		if (k <= 0 || k > totalNum) {
			return null;
		}
		int perNum = totalNum / n;
		int startNum = (k - 1) / perNum;
		int[] array = new int[n];
		for (int i = 0; i < n; i++) {
			array[i] = i + 1;
		}
		int temp = array[startNum];
		for (int i = startNum; i > 0; i--) {
			array[i] = array[i - 1];
		}
		array[0] = temp;
		int permutationNum = (k - 1) % perNum;
		for (int i = 0; i < permutationNum; i++) {
			nextPermutation(array, 0, array.length - 1);
		}

		String result = "";
		for (int i = 0; i < array.length; i++) {
			result += String.valueOf(array[i]);
		}
		return result;
	}

	/**
	 * $(Next Permutation)
	 * 
	 * Implement next permutation, which rearranges numbers into the
	 * lexicographically next greater permutation of numbers.
	 * 
	 * If such arrangement is not possible, it must rearrange it as the lowest
	 * possible order (ie, sorted in ascending order).
	 * 
	 * The replacement must be in-place, do not allocate extra memory.
	 * 
	 * Here are some examples. Inputs are in the left-hand column and its
	 * corresponding outputs are in the right-hand column. 1,2,3 → 1,3,2 3,2,1 →
	 * 1,2,3 1,1,5 → 1,5,1
	 */

	public void nextPermutation(int[] num) {
		nextPermutation(num, 0, num.length - 1);
	}

	/**
	 * $(Permutations II)
	 * 
	 * Given a collection of numbers that might contain duplicates, return all
	 * possible unique permutations.
	 * 
	 * For example, [1,1,2] have the following unique permutations: [1,1,2],
	 * [1,2,1], and [2,1,1].
	 */

	/*
	 * add num[i] to array only if i is not used and the previous duplicated
	 * element is used.
	 */
	public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null) {
			return result;
		}

		Arrays.sort(num);
		ArrayList<Integer> permutation = new ArrayList<Integer>();
		boolean[] visited = new boolean[num.length];
		permuteUniqueDFS(result, permutation, num, visited);

		return result;
	}

	private void permuteUniqueDFS(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> permutation, int[] num, boolean[] visited) {
		if (permutation.size() == num.length) {
			result.add(new ArrayList<Integer>(permutation));
			return;
		}

		for (int i = 0; i < num.length; i++) {
			if (visited[i] || i > 0 && num[i] == num[i - 1] && !visited[i - 1]) {
				continue;
			}
			visited[i] = true;
			permutation.add(num[i]);
			permuteUniqueDFS(result, permutation, num, visited);
			permutation.remove(permutation.size() - 1);
			visited[i] = false;
		}
	}

	/**
	 * $(Convert Sorted Array to Binary Search Tree)
	 * 
	 * Given an array where elements are sorted in ascending order, convert it
	 * to a height balanced BST.
	 */

	public TreeNode sortedArrayToBST(int[] num) {
		if (num == null || num.length == 0) {
			return null;
		}

		return sortedArrayToBST(num, 0, num.length - 1);
	}

	public TreeNode sortedArrayToBST(int[] num, int start, int end) {
		if (start > end) {
			return null;
		} else if (start == end) {
			return new TreeNode(num[start]);
		}

		int mid = start + (end - start) / 2;
		TreeNode root = new TreeNode(num[mid]);
		root.left = sortedArrayToBST(num, start, mid - 1);
		root.right = sortedArrayToBST(num, mid + 1, end);
		return root;
	}

	private void printBST(TreeNode root) {
		if (root == null) {
			return;
		}
		printBST(root.left);
		System.out.println(root.val);
		printBST(root.right);
	}

	@Test
	public void testSortedArrayToBST() {
		int[] array = { 1, 2, 3, 4, 5, 6, 7, };
		TreeNode root = sortedArrayToBST(array);
		printBST(root);
	}

	/**
	 * $(Binary Tree Preorder Traversal )
	 * 
	 * Given a binary tree, return the preorder traversal of its nodes' values.
	 * 
	 * Note: Recursive solution is trivial, could you do it iteratively?
	 */

	public ArrayList<Integer> preorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode curr = root;
		boolean stop = false;
		while (!stop) {
			if (curr != null) {
				result.add(curr.val);
				stack.push(curr);
				curr = curr.left;
			} else {
				if (!stack.isEmpty()) {
					curr = stack.pop().right;
				} else {
					stop = true;
				}
			}
		}
		return result;
	}

	/**
	 * $(Binary Tree Inorder Traversal)
	 * 
	 * Given a binary tree, return the inorder traversal of its nodes' values.
	 * Note: Recursive solution is trivial, could you do it iteratively?
	 */

	public ArrayList<Integer> inorderTraversalRecursive(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		inorderTraversalRecursive(root, result);
		return result;
	}

	private void inorderTraversalRecursive(TreeNode root,
			ArrayList<Integer> result) {
		if (root == null) {
			return;
		}
		inorderTraversalRecursive(root.left, result);
		result.add(root.val);
		inorderTraversalRecursive(root.right, result);
	}

	public ArrayList<Integer> inorderTraversalIterative(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode curr = root;
		boolean stop = false;
		while (!stop) {
			if (curr != null) {
				stack.push(curr);
				curr = curr.left;
			} else {
				if (stack.isEmpty()) {
					stop = true;
				} else {
					curr = stack.pop();
					result.add(curr.val);
					curr = curr.right;
				}
			}
		}
		return result;
	}

	/**
	 * $(Swap Nodes in Pairs)
	 * 
	 * Given a linked list, swap every two adjacent nodes and return its head.
	 * 
	 * For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
	 * 
	 * Your algorithm should use only constant space. You may not modify the
	 * values in the list, only nodes itself can be changed.
	 */

	public ListNode swapPairs(ListNode head) {
		if (head == null) {
			return head;
		}
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode front = head.next, mid = head, back = dummy;
		while (front != null) {
			mid.next = front.next;
			front.next = mid;
			back.next = front;
			back = mid;
			mid = mid.next;
			if (mid == null) {
				break;
			}
			front = mid.next;
		}
		return dummy.next;
	}

	/**
	 * $(Pascal's Triangle)
	 * 
	 * Given numRows, generate the first numRows of Pascal's triangle.
	 * 
	 * For example, given numRows = 5, Return
	 * 
	 * [ [1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1] ]
	 */

	public ArrayList<ArrayList<Integer>> generate(int numRows) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (numRows == 0) {
			return result;
		}

		ArrayList<Integer> lastRow = new ArrayList<Integer>();
		lastRow.add(1);
		result.add(lastRow);
		int row = 2;
		while (row <= numRows) {
			ArrayList<Integer> newRow = new ArrayList<Integer>();
			newRow.add(1);
			for (int i = 0; i < row - 2; i++) {
				newRow.add(lastRow.get(i) + lastRow.get(i + 1));
			}
			newRow.add(1);
			result.add(newRow);
			lastRow = newRow;
			row++;
		}

		return result;
	}

	/**
	 * $(Best Time to Buy and Sell Stock)
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i.
	 * 
	 * If you were only permitted to complete at most one transaction (ie, buy
	 * one and sell one share of the stock), design an algorithm to find the
	 * maximum profit.
	 */

	/*
	 * Brute force, O(n^2), can be improved by finding all the lowest and
	 * highest days first, then apply brute force to the reduced array.
	 * Improvement is not guaranteed, the worst case is still O(n^2) if the
	 * everyday is a lowest or highest.
	 */
	public int maxProfitBruteForce(int[] prices) {
		int maxProfit = 0;
		if (prices != null) {
			for (int i = 0; i < prices.length - 1; i++) {
				Integer max = null;
				for (int j = i + 1; j < prices.length; j++) {
					if (prices[j] > prices[i]) {
						if (max == null || prices[j] > max) {
							max = prices[j];
						}
					}
				}
				int profit = max - prices[i];
				maxProfit = Math.max(profit, maxProfit);
			}
		}

		return maxProfit;
	}

	public int maxProfit(int[] prices) {
		if (prices == null || prices.length < 2) {
			return 0;
		}

		int len = prices.length, low = prices[0], maxProfit = 0;
		for (int i = 1; i < len; i++) {
			low = Math.min(low, prices[i]);
			maxProfit = Math.max(maxProfit, prices[i] - low);
		}

		return maxProfit;
	}

	/**
	 * $(Integer to Roman)
	 * 
	 * Given an integer, convert it to a roman numeral.
	 * 
	 * Input is guaranteed to be within the range from 1 to 3999.
	 * 
	 * M 1000 D 500 C 100 L 50 X 10 V 5 I 1
	 */

	public String intToRoman(int num) {
		int[] weight = { 1000, 500, 100, 50, 10, 5, 1 };
		char[] roman = { 'M', 'D', 'C', 'L', 'X', 'V', 'I' };
		int[] m = new int[7];
		for (int i = 0; i < 7; i++) {
			m[i] = num / weight[i];
			num %= weight[i];
		}
		StringBuilder result = new StringBuilder();
		int index = 6;
		while (index >= 0) {
			num = m[index];
			if (num == 4) {
				if (m[index - 1] == 0) {
					result.insert(0, roman[index - 1]);
				} else {
					result.insert(0, roman[index - 2]);
				}
				result.insert(0, roman[index]);
				index -= 2;
			} else {
				for (int i = 0; i < num; i++) {
					result.insert(0, roman[index]);
				}
				index--;
			}
		}
		return result.toString();
	}

	/**
	 * $(Balanced Binary Tree)
	 * 
	 * Given a binary tree, determine if it is height-balanced.
	 * 
	 * For this problem, a height-balanced binary tree is defined as a binary
	 * tree in which the depth of the two subtrees of every node never differ by
	 * more than 1.
	 */

	public boolean isBalanced(TreeNode root) {
		if (root == null) {
			return true;
		}

		return isBalanced(root.left) && isBalanced(root.right)
				&& Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1;
	}

	/**
	 * $(Binary Tree Level Order Traversal )
	 * 
	 * Given a binary tree, return the level order traversal of its nodes'
	 * values. (ie, from left to right, level by level).
	 * 
	 * For example: Given binary tree {3,9,20,#,#,15,7} return its level order
	 * traversal as: [ [3], [9,20], [15,7] ]
	 */

	public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (root == null) {
			return result;
		}

		ArrayList<TreeNode> last = new ArrayList<TreeNode>();
		last.add(root);
		while (!last.isEmpty()) {
			ArrayList<Integer> layer = new ArrayList<Integer>();
			ArrayList<TreeNode> next = new ArrayList<TreeNode>();
			for (TreeNode node : last) {
				layer.add(node.val);
				if (node.left != null) {
					next.add(node.left);
				}
				if (node.right != null) {
					next.add(node.right);
				}
			}
			result.add(layer);
			last = next;
		}

		return result;
	}

	/**
	 * $(Binary Tree Level Order Traversal II )
	 * 
	 * Given a binary tree, return the bottom-up level order traversal of its
	 * nodes' values. (ie, from left to right, level by level from leaf to
	 * root).
	 * 
	 * For example: Given binary tree {3,9,20,#,#,15,7}, return its bottom-up
	 * level order traversal as: [ [15,7] [9,20], [3], ]
	 */

	public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
		ArrayList<ArrayList<Integer>> topOrder = levelOrder(root);
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		int len = topOrder.size();
		for (int i = 0; i < len; i++) {
			result.add(topOrder.get(len - i - 1));
		}
		return result;
	}

	/**
	 * $(Single Number II)
	 * 
	 * Given an array of integers, every element appears three times except for
	 * one. Find that single one.
	 * 
	 * Note: Your algorithm should have a linear runtime complexity. Could you
	 * implement it without using extra memory?
	 */

	public int singleNumber2(int[] A) {
		int[] count = new int[32];
		int result = 0;
		for (int i = 0; i < count.length; i++) {
			int mask = 1 << i;
			for (int j = 0; j < A.length; j++) {
				if ((A[j] & mask) != 0) {
					count[i] = (count[i] + 1) % 3;
				}
			}
			result |= count[i] << i;
		}

		return result;
	}

	/**
	 * $(Unique Paths)
	 * 
	 * A robot is located at the top-left corner of a m x n grid.
	 * 
	 * The robot can only move either down or right at any point in time. The
	 * robot is trying to reach the bottom-right corner of the grid.
	 * 
	 * How many possible unique paths are there?
	 * 
	 * Note: m and n will be at most 100.
	 */

	public int uniquePaths(int m, int n) {
		if (m <= 0 || n <= 0) {
			return 0;
		}

		int[][] dp = new int[m][n];
		for (int r = 0; r < m; r++) {
			dp[r][n - 1] = 1;
		}
		for (int c = 0; c < n; c++) {
			dp[m - 1][c] = 1;
		}
		for (int r = m - 2; r >= 0; r--) {
			for (int c = n - 2; c >= 0; c--) {
				dp[r][c] = dp[r + 1][c] + dp[r][c + 1];
			}
		}

		return dp[0][0];
	}

	/**
	 * $(Minimum Path Sum)
	 * 
	 * Given a m x n grid filled with non-negative numbers, find a path from top
	 * left to bottom right which minimizes the sum of all numbers along its
	 * path.
	 * 
	 * Note: You can only move either down or right at any point in time.
	 */

	public int minPathSum(int[][] grid) {
		if (grid == null || grid.length == 0 || grid[0] == null
				|| grid[0].length == 0) {
			return 0;
		}

		int height = grid.length, width = grid[0].length;
		int[][] dp = new int[height][width];
		dp[height - 1][width - 1] = grid[height - 1][width - 1];
		for (int col = width - 2; col >= 0; col--) {
			dp[height - 1][col] = grid[height - 1][col]
					+ dp[height - 1][col + 1];
		}
		for (int row = height - 2; row >= 0; row--) {
			dp[row][width - 1] = grid[row][width - 1] + dp[row + 1][width - 1];
		}

		for (int row = height - 2; row >= 0; row--) {
			for (int col = width - 2; col >= 0; col--) {
				dp[row][col] = grid[row][col]
						+ Math.min(dp[row + 1][col], dp[row][col + 1]);
			}
		}

		return dp[0][0];
	}

	/**
	 * $(Spiral Matrix II)
	 * 
	 * Given an integer n, generate a square matrix filled with elements from 1
	 * to n^2 in spiral order.
	 * 
	 * For example, Given n = 3,
	 * 
	 * You should return the following matrix:
	 * 
	 * [ [ 1, 2, 3 ], [ 8, 9, 4 ], [ 7, 6, 5 ] ]
	 */

	public int[][] generateMatrix(int n) {
		if (n == 0) {
			return new int[][] {};
		}
		int[][] directions = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
		final int right = 0, down = 1, left = 2, up = 3, row = 0, col = 1;
		int state = right;
		int[][] m = new int[n][n];
		int num = n * n;
		int index = 1;
		m[0][0] = index++;
		int r = 0, c = 0;
		for (int i = 1; i < num; i++) {
			switch (state) {
			case right:
				if (needTurn(r, c + 1, n, m)) {
					state = (state + 1) % 4;
				}
				break;
			case down:
				if (needTurn(r + 1, c, n, m)) {
					state = (state + 1) % 4;
				}
				break;
			case left:
				if (needTurn(r, c - 1, n, m)) {
					state = (state + 1) % 4;
				}
				break;
			case up:
				if (needTurn(r - 1, c, n, m)) {
					state = (state + 1) % 4;
				}
				break;
			}
			r += directions[state][row];
			c += directions[state][col];
			m[r][c] = index++;
		}

		return m;
	}

	private boolean needTurn(int row, int col, int n, int[][] m) {
		if (row < 0 || row > n - 1 || col < 0 || col > n - 1
				|| m[row][col] != 0) {
			return true;
		}
		return false;
	}

	@Test
	public void testGenerateMatrix() {
		int m[][] = generateMatrix(0);
		for (int[] arr : m) {
			System.out.println(Arrays.toString(arr));
		}
	}

	/**
	 * $(Container With Most Water)
	 * 
	 * Given n non-negative integers a1, a2, ..., an, where each represents a
	 * point at coordinate (i, ai). n vertical lines are drawn such that the two
	 * end points of line i is at (i, ai) and (i, 0). Find two lines, which
	 * together with x-axis forms a container, such that the container contains
	 * the most water.
	 * 
	 * Note: You may not slant the container.
	 */

	/*
	 * Greedy solution: set two pointers, one at each end, each time the pointer
	 * which points to the shorter end moves inward by one, because the area
	 * will only reduce by moving the pointer at longer end.
	 */

	public int maxArea(int[] height) {
		if (height == null || height.length < 2) {
			return 0;
		}

		int len = height.length, left = 0, right = len - 1, maxArea = 0;
		while (left < right) {
			if (height[left] < height[right]) {
				maxArea = Math.max(maxArea, height[left] * (right - left));
				left++;
			} else {
				maxArea = Math.max(maxArea, height[right] * (right - left));
				right--;
			}
		}

		return maxArea;
	}

	/**
	 * $(Rotate Image)
	 * 
	 * You are given an n x n 2D matrix representing an image.
	 * 
	 * Rotate the image by 90 degrees (clockwise).
	 * 
	 * Follow up: Could you do this in-place?
	 */

	public void rotate(int[][] matrix) {
		if (matrix == null) {
			return;
		}
		int n = matrix.length;
		for (int r = 0; r < n / 2; r++) {
			for (int c = 0; c < n; c++) {
				matrix[r][c] ^= matrix[n - 1 - r][c];
				matrix[n - 1 - r][c] ^= matrix[r][c];
				matrix[r][c] ^= matrix[n - 1 - r][c];
			}
		}
		for (int r = 1; r < n; r++) {
			for (int c = 0; c < r; c++) {
				matrix[r][c] ^= matrix[c][r];
				matrix[c][r] ^= matrix[r][c];
				matrix[r][c] ^= matrix[c][r];
			}
		}
	}

	public void rotate2(int[][] matrix) {
		int len = matrix.length;
		for (int r = 0; r < len / 2; r++) {
			for (int c = r; c < len - 1 - r; c++) {
				int r1 = r, c1 = c, temp = matrix[r1][c1];
				int r2 = c1, c2 = len - 1 - r1;
				int r3 = c2, c3 = len - 1 - r2;
				int r4 = c3, c4 = len - 1 - r3;
				matrix[r1][c1] = matrix[r4][c4];
				matrix[r4][c4] = matrix[r3][c3];
				matrix[r3][c3] = matrix[r2][c2];
				matrix[r2][c2] = temp;
			}
		}
	}

	/**
	 * $(Generate Parentheses)
	 * 
	 * Given n pairs of parentheses, write a function to generate all
	 * combinations of well-formed parentheses.
	 * 
	 * For example, given n = 3, a solution set is:
	 * 
	 * "((()))", "(()())", "(())()", "()(())", "()()()"
	 */

	public ArrayList<String> generateParenthesis(int n) {
		ArrayList<String> result = new ArrayList<String>();
		if (n > 0) {
			generateParenthesisDFS(result, "", n, 0);
		}
		return result;
	}

	public void generateParenthesisDFS(ArrayList<String> result, String sofar,
			int toOpen, int toClose) {
		if (toClose == 0 && toOpen == 0) {
			result.add(sofar);
			return;
		}

		if (toClose > 0) {
			generateParenthesisDFS(result, sofar + ")", toOpen, toClose - 1);
		}

		if (toOpen > 0) {
			generateParenthesisDFS(result, sofar + "(", toOpen - 1, toClose + 1);
		}
	}

	/**
	 * $(Pascal's Triangle II)
	 * 
	 * Given an index k, return the kth row of the Pascal's triangle.
	 * 
	 * For example, given k = 3, Return [1,3,3,1].
	 * 
	 * Note: Could you optimize your algorithm to use only O(k) extra space?
	 */

	public ArrayList<Integer> getRow(int rowIndex) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (rowIndex >= 0) {
			result.add(1);
			for (int i = 1; i <= rowIndex; i++) {
				ArrayList<Integer> newArr = new ArrayList<Integer>();
				newArr.add(1);
				for (int j = 1; j <= i - 1; j++) {
					newArr.add(result.get(j) + result.get(j - 1));
				}
				newArr.add(1);
				result = newArr;
			}
		}
		return result;
	}

	@Test
	public void testGetRow() {
		for (int i = -1; i < 10; i++) {
			ArrayList<Integer> result = getRow(i);
			System.out.println(i + ":" + result);
		}
	}

	/**
	 * $(Plus One)
	 * 
	 * Given a number represented as an array of digits, plus one to the number.
	 */

	public int[] plusOne(int[] digits) {
		if (digits == null) {
			return null;
		}

		int len = digits.length, carry = 1, ptr = len - 1;
		while (ptr >= 0 && carry == 1) {
			int sum = digits[ptr] + carry;
			if (sum >= 10) {
				sum -= 10;
				carry = 1;
			} else {
				carry = 0;
			}
			digits[ptr] = sum;
			ptr--;
		}

		if (carry == 1) {
			int[] result = new int[len + 1];
			result[0] = carry;
			for (int i = 1; i < result.length; i++) {
				result[i] = digits[i - 1];
			}
			return result;
		} else {
			return digits;
		}
	}

	/**
	 * $(Remove Nth Node From End of List)
	 * 
	 * Given a linked list, remove the nth node from the end of list and return
	 * its head.
	 * 
	 * For example,
	 * 
	 * Given linked list: 1->2->3->4->5, and n = 2.
	 * 
	 * After removing the second node from the end, the linked list becomes
	 * 1->2->3->5. Note: Given n will always be valid. Try to do this in one
	 * pass.
	 */

	public ListNode removeNthFromEnd(ListNode head, int n) {
		if (head == null || n <= 0) {
			return null;
		}

		ListNode dummy = new ListNode(0), front = dummy, back = dummy;
		dummy.next = head;
		while (n >= 0) {
			front = front.next;
			n--;
		}

		while (front != null) {
			front = front.next;
			back = back.next;
		}

		back.next = back.next.next;

		return dummy.next;
	}

	/**
	 * $(Set Matrix Zeroes)
	 * 
	 * Given a m * n matrix, if an element is 0, set its entire row and column
	 * to 0. Do it in place.
	 * 
	 * Follow up: Did you use extra space? A straight forward solution using
	 * O(mn) space is probably a bad idea. A simple improvement uses O(m + n)
	 * space, but still not the best solution. Could you devise a constant space
	 * solution?
	 */

	public void setZeroes(int[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null
				|| matrix[0].length == 0) {
			return;
		}

		int h = matrix.length, w = matrix[0].length;
		boolean clearFirstRow = false;
		for (int c = 0; c < w; c++) {
			if (matrix[0][c] == 0) {
				clearFirstRow = true;
				break;
			}
		}

		for (int r = 1; r < h; r++) {
			boolean clear = false;
			for (int c = 0; c < w; c++) {
				if (matrix[r][c] == 0) {
					clear = true;
					matrix[0][c] = 0;
				}
			}
			if (clear) {
				for (int c = 0; c < w; c++) {
					matrix[r][c] = 0;
				}
			}
		}

		for (int c = 0; c < w; c++) {
			if (matrix[0][c] == 0) {
				for (int r = 0; r < h; r++) {
					matrix[r][c] = 0;
				}
			}
		}

		if (clearFirstRow) {
			for (int c = 0; c < w; c++) {
				matrix[0][c] = 0;
			}
		}
	}

	/**
	 * $(Sort Colors)
	 * 
	 * Given an array with n objects colored red, white or blue, sort them so
	 * that objects of the same color are adjacent, with the colors in the order
	 * red, white and blue.
	 * 
	 * Here, we will use the integers 0, 1, and 2 to represent the color red,
	 * white, and blue respectively.
	 * 
	 * Follow up: A rather straight forward solution is a two-pass algorithm
	 * using counting sort. First, iterate the array counting number of 0's,
	 * 1's, and 2's, then overwrite array with total number of 0's, then 1's and
	 * followed by 2's.
	 * 
	 * Could you come up with an one-pass algorithm using only constant space?
	 */

	public void sortColors(int[] A) {
		if (A != null && A.length > 1) {
			int len = A.length, lp = 0, rp = len - 1, p = 0;
			while (p <= rp) {
				if (A[p] == 0) {
					arraySwap(A, lp, p);
					lp++;
					p++;
				} else if (A[p] == 2) {
					arraySwap(A, rp, p);
					rp--;
				} else {
					p++;
				}
			}
		}
	}

	public boolean nextNBand(int[] a, int band) {
		if (a == null) {
			throw new RuntimeException("Null array");
		}
		if (a.length == 0) {
			return false;
		}
		int len = a.length;
		for (int i = 0; i < len; i++) {
			if (a[i] >= band || a[i] < 0) {
				throw new RuntimeException("Invalid input");
			}
		}
		int index = len - 1;
		a[index] += 1;
		while (a[index] == band) {
			a[index--] = 0;
			if (index >= 0) {
				a[index] += 1;
			} else {
				break;
			}
		}
		if (index == -1) {
			return false;
		}

		return true;
	}

	/**
	 * $(Path Sum)
	 * 
	 * Given a binary tree and a sum, determine if the tree has a root-to-leaf
	 * path such that adding up all the values along the path equals the given
	 * sum.
	 */

	public boolean hasPathSum(TreeNode root, int sum) {
		if (root == null) {
			return false;
		}

		return hasPathSum(root, 0, sum);
	}

	public boolean hasPathSum(TreeNode root, int sofar, int target) {
		if (root.left == null && root.right == null) {
			return root.val + sofar == target;
		} else if (root.left != null && root.right == null) {
			return hasPathSum(root.left, sofar + root.val, target);
		} else if (root.left == null && root.right != null) {
			return hasPathSum(root.right, sofar + root.val, target);
		} else {
			if (!hasPathSum(root.left, sofar + root.val, target)) {
				return hasPathSum(root.right, sofar + root.val, target);
			} else {
				return true;
			}
		}
	}

	/**
	 * $(Linked List Cycle)
	 * 
	 * Given a linked list, determine if it has a cycle in it.
	 * 
	 * Follow up: Can you solve it without using extra space?
	 */

	/*
	 * 1. In a cycled singly linked list none of the nodes points to null as its
	 * next.
	 * 
	 * 2. Use two pointers to iterate the linked list, one faster than the
	 * other, if there is a cycle, they will finally meet in the cycle.
	 */
	public boolean hasCycle(ListNode head) {
		ListNode slow = head, fast = head;
		while (fast != null) {
			fast = fast.next;
			if (fast == null) {
				return false;
			}
			fast = fast.next;
			slow = slow.next;
			if (fast == slow) {
				return true;
			}
		}

		return false;
	}

	/**
	 * $(Linked List Cycle II)
	 * 
	 * Given a linked list, return the node where the cycle begins. If there is
	 * no cycle, return null.
	 * 
	 * Follow up: Can you solve it without using extra space?
	 */

	public ListNode detectCycle(ListNode head) {
		ListNode fast = head, slow = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (slow == fast) {
				break;
			}
		}

		if (fast == null || fast.next == null) {
			return null;
		}

		slow = head;
		while (slow != fast) {
			fast = fast.next;
			slow = slow.next;
		}
		return fast;
	}

	/*
	 * use hashset to store visited nodes
	 */
	public ListNode detectCycle2(ListNode head) {
		if (head == null) {
			return head;
		}

		Set<ListNode> vst = new HashSet<ListNode>();
		ListNode ptr = head;
		while (ptr != null) {
			if (vst.contains(ptr)) {
				return ptr;
			}
			vst.add(ptr);
			ptr = ptr.next;
		}

		return null;
	}

	/**
	 * $(Combinations)
	 * 
	 * Given two integers n and k, return all possible combinations of k numbers
	 * out of 1 ... n.
	 * 
	 * For example, If n = 4 and k = 2, a solution is:
	 * 
	 * [ [2,4], [3,4], [2,3], [1,2], [1,3], [1,4], ]
	 */

	public ArrayList<ArrayList<Integer>> combine(int n, int k) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (n < 0 || k < 0 || n < k) {
			return result;
		}

		ArrayList<Integer> combination = new ArrayList<Integer>();
		combineDFS(result, 0, n, k, combination);

		return result;
	}

	private void combineDFS(ArrayList<ArrayList<Integer>> result, int curr,
			int n, int k, ArrayList<Integer> combination) {
		if (combination.size() == k) {
			result.add(new ArrayList<Integer>(combination));
			return;
		}

		for (int i = curr; i <= n - k + combination.size(); i++) {
			combination.add(i + 1);
			combineDFS(result, i + 1, n, k, combination);
			combination.remove(combination.size() - 1);
		}
	}

	/**
	 * $(Remove Duplicates from Sorted Array II)
	 * 
	 * Follow up for "Remove Duplicates": What if duplicates are allowed at most
	 * twice?
	 * 
	 * For example, Given sorted array A = [1,1,1,2,2,3],
	 * 
	 * Your function should return length = 5, and A is now [1,1,2,2,3].
	 */

	public int removeDuplicates2(int[] A) {
		if (A == null || A.length == 0) {
			return 0;
		}
		int top = 1;
		boolean allow = true;
		for (int ptr = 1; ptr < A.length; ptr++) {
			if (A[ptr] == A[ptr - 1]) {
				if (allow) {
					allow = false;
					A[top++] = A[ptr];
				}
			} else {
				A[top++] = A[ptr];
				allow = true;
			}
		}

		return top;
	}

	@Test
	public void testRemoveDuplicates2() {
		int[] A = { 1, 2, 3, 4, 5, };
		System.out.println(removeDuplicates2(A));
		System.out.println(Arrays.toString(A));
	}

	/**
	 * $(Subsets)
	 * 
	 * Given a set of distinct integers, S, return all possible subsets.
	 * 
	 * Note: Elements in a subset must be in non-descending order. The solution
	 * set must not contain duplicate subsets. For example, If S = [1,2,3], a
	 * solution is:
	 * 
	 * [ [3], [1], [2], [1,2,3], [1,3], [2,3], [1,2], [] ]
	 */

	public ArrayList<ArrayList<Integer>> subsets(int[] S) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (S == null) {
			return result;
		}

		Arrays.sort(S);
		ArrayList<Integer> combination = new ArrayList<Integer>();
		subsetsDFS(result, combination, S, 0);

		return result;
	}

	private void subsetsDFS(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> combination, int[] S, int pos) {
		if (pos == S.length) {
			result.add(new ArrayList<Integer>(combination));
			return;
		}

		combination.add(S[pos]);
		subsetsDFS(result, combination, S, pos + 1);
		combination.remove(combination.size() - 1);
		subsetsDFS(result, combination, S, pos + 1);
	}

	/**
	 * $(Palindrome Number)
	 * 
	 * Determine whether an integer is a palindrome. Do this without extra
	 * space.
	 */

	public boolean isPalindrome(int x) {
		if (x < 0) {
			return false;
		}
		int div = 1;
		while (x / div >= 10) {
			div *= 10;
		}
		while (x > 0) {
			int l = x / div;
			int r = x % 10;
			if (l != r) {
				return false;
			}
			x = (x % div) / 10;
			div /= 100;
		}
		return true;
	}

	/**
	 * $(Reorder List)
	 * 
	 * Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it to:
	 * L0→Ln→L1→Ln-1→L2→Ln-2→...
	 * 
	 * You must do this in-place without altering the nodes' values.
	 * 
	 * For example, Given {1,2,3,4}, reorder it to {1,4,2,3}.
	 */

	public void reorderList(ListNode head) {
		if (head == null) {
			return;
		}

		ListNode dummy = new ListNode(0), fast = dummy, slow = dummy;
		dummy.next = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
		}

		ListNode dummy2 = new ListNode(0);
		dummy2.next = reverseLinkedList(slow.next);
		slow.next = null;

		ListNode ptr1 = head, ptr2 = dummy2.next;
		while (ptr2 != null) {
			dummy2.next = ptr2.next;
			ptr2.next = ptr1.next;
			ptr1.next = ptr2;
			ptr1 = ptr2.next;
			ptr2 = dummy2.next;
		}
	}

	private ListNode reverseLinkedList(ListNode head) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode ptr = dummy.next, newHead = null;
		while (ptr != null) {
			dummy.next = ptr.next;
			ptr.next = newHead;
			newHead = ptr;
			ptr = dummy.next;
		}

		return newHead;
	}

	/**
	 * $(Populating Next Right Pointers in Each Node II)
	 * 
	 * Follow up for problem "Populating Next Right Pointers in Each Node".
	 * 
	 * What if the given tree could be any binary tree? Would your previous
	 * solution still work?
	 * 
	 * Note:
	 * 
	 * You may only use constant extra space.
	 */

	public void connect2(TreeLinkNode root) {
		TreeLinkNode start = root;
		while (start != null) {
			TreeLinkNode ptr = start;
			while (ptr != null) {
				TreeLinkNode child = null;
				if (ptr.left != null && ptr.right != null) {
					ptr.left.next = ptr.right;
					child = ptr.right;
				} else if (ptr.left != null) {
					child = ptr.left;
				} else if (ptr.right != null) {
					child = ptr.right;
				}
				if (child != null) {
					TreeLinkNode sibling = null;
					while (true) {
						ptr = ptr.next;
						if (ptr == null) {
							break;
						}
						if (ptr.left != null) {
							sibling = ptr.left;
							break;
						} else if (ptr.right != null) {
							sibling = ptr.right;
							break;
						}
					}
					child.next = sibling;
				} else {
					ptr = ptr.next;
				}
			}
			TreeLinkNode newStart = null;
			ptr = start;
			while (ptr != null) {
				if (ptr.left != null) {
					newStart = ptr.left;
					break;
				} else if (ptr.right != null) {
					newStart = ptr.right;
					break;
				}
				ptr = ptr.next;
			}
			start = newStart;
		}
	}

	private void printLinkedListTree(TreeLinkNode root) {
		if (root == null) {
			System.out.println("null");
		}
		TreeLinkNode start = root;
		while (start != null) {
			TreeLinkNode ptr = start;
			while (ptr != null) {
				System.out.print(ptr.val + " ");
				ptr = ptr.next;
			}
			System.out.println();
			ptr = start;
			TreeLinkNode firstChild = null;
			while (ptr != null) {
				if (ptr.left != null) {
					firstChild = ptr.left;
					break;
				} else if (ptr.right != null) {
					firstChild = ptr.right;
					break;
				} else {
					ptr = ptr.next;
				}
			}
			start = firstChild;
		}
	}

	@Test
	public void testConnect2() {
		TreeLinkNode root = new TreeLinkNode(0);
		TreeLinkNode l1 = new TreeLinkNode(1);
		root.left = l1;
		TreeLinkNode l2 = new TreeLinkNode(2);
		l1.left = l2;
		TreeLinkNode r1 = new TreeLinkNode(3);
		root.right = r1;
		TreeLinkNode r2 = new TreeLinkNode(4);
		r1.right = r2;
		l2.left = new TreeLinkNode(9);
		l2.right = new TreeLinkNode(7);
		r2.left = new TreeLinkNode(8);
		connect2(root);
		printLinkedListTree(root);
	}

	/**
	 * $(Copy List with Random Pointer_
	 * 
	 * A linked list is given such that each node contains an additional random
	 * pointer which could point to any node in the list or null.
	 * 
	 * Return a deep copy of the list.
	 */

	class RandomListNode {
		int label;
		RandomListNode next, random;

		RandomListNode(int x) {
			this.label = x;
		}
	};

	/*
	 * use a hashmap old node -> new node
	 */
	public RandomListNode copyRandomList(RandomListNode head) {
		RandomListNode dummy = new RandomListNode(0), cpyptr = dummy, ptr = head;
		Map<RandomListNode, RandomListNode> map = new HashMap<RandomListNode, RandomListNode>();

		while (ptr != null) {
			RandomListNode newNode = new RandomListNode(ptr.label);
			map.put(ptr, newNode);
			cpyptr.next = newNode;
			cpyptr = cpyptr.next;
			ptr = ptr.next;
		}

		for (RandomListNode node : map.keySet()) {
			map.get(node).random = map.get(node.random);
		}

		return dummy.next;
	}

	/*
	 * without hashmap
	 */
	public RandomListNode copyRandomList2(RandomListNode head) {
		if (head == null) {
			return null;
		}

		copyNext(head);
		copyRandom(head);
		return split(head);
	}

	private void copyNext(RandomListNode head) {
		RandomListNode ptr = head;
		while (ptr != null) {
			RandomListNode newNode = new RandomListNode(ptr.label);
			newNode.next = ptr.next;
			ptr.next = newNode;
			ptr = ptr.next.next;
		}
	}

	private void copyRandom(RandomListNode head) {
		RandomListNode ptr = head;
		while (ptr != null) {
			if (ptr.random != null) {
				ptr.next.random = ptr.random.next;
			} else {
				ptr.next.random = null;
			}
			ptr = ptr.next.next;
		}
	}

	private RandomListNode split(RandomListNode head) {
		if (head == null) {
			return null;
		}
		RandomListNode newHead = head.next, ptr = head, cpyptr = newHead;
		while (ptr != null) {
			ptr.next = cpyptr.next;
			ptr = ptr.next;
			if (cpyptr.next != null) {
				cpyptr.next = cpyptr.next.next;
			} else {
				cpyptr.next = null;
			}
			cpyptr = cpyptr.next;
		}

		return newHead;
	}

	/**
	 * $(Restore IP Addresses)
	 * 
	 * Given a string containing only digits, restore it by returning all
	 * possible valid IP address combinations.
	 * 
	 * For example: Given "25525511135",
	 * 
	 * return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
	 */

	public ArrayList<String> restoreIpAddresses(String s) {
		ArrayList<String> result = new ArrayList<String>();
		if (s == null || !s.matches("\\d{4,12}")) {
			return result;
		}

		ArrayList<String> combination = new ArrayList<String>();
		restoreIpAddressesDFS(result, combination, s, 3);

		return result;
	}

	private void restoreIpAddressesDFS(ArrayList<String> result,
			ArrayList<String> combination, String s, int seg) {
		if (seg == 0) {
			if (!s.isEmpty() && Integer.parseInt(s) <= 255
					&& (s.charAt(0) != '0' || s.length() == 1)) {
				String ip = "";
				for (int i = 0; i < combination.size(); i++) {
					ip += combination.get(i) + ".";
				}
				ip += s;
				result.add(ip);
			}
			return;
		}

		int maxDigit = 1;
		if (s.charAt(0) != '0') {
			maxDigit = Math.min(3, s.length());
		}
		String sofar = "";
		for (int i = 0; i < maxDigit; i++) {
			sofar += s.charAt(i);
			int remain = (s.length() - i - 1) / seg;
			if (Integer.parseInt(sofar) <= 255 && remain <= 3 && remain >= 1) {
				combination.add(sofar);
				restoreIpAddressesDFS(result, combination, s.substring(i + 1),
						seg - 1);
				combination.remove(combination.size() - 1);
			}
		}
	}

	/**
	 * $(Binary Tree Postorder Traversal)
	 * 
	 * Given a binary tree, return the postorder traversal of its nodes' values.
	 * 
	 * Note: Recursive solution is trivial, could you do it iteratively?
	 */

	public ArrayList<Integer> postorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		TreeNode curr = root, prev = null;
		Stack<TreeNode> stack = new Stack<TreeNode>();
		boolean stop = false;

		while (!stop) {
			if (curr != null) {
				stack.push(curr);
				prev = curr;
				curr = curr.left;
			} else {
				if (!stack.isEmpty()) {
					curr = stack.pop();
					if (curr.right == null || prev == curr.right) {
						result.add(curr.val);
						prev = curr;
						curr = null;
					} else {
						stack.push(curr);
						prev = curr;
						curr = curr.right;
					}
				} else {
					stop = true;
				}
			}
		}

		return result;
	}

	/**
	 * $(Search in Rotated Sorted Array)
	 * 
	 * Suppose a sorted array is rotated at some pivot unknown to you
	 * beforehand.
	 * 
	 * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
	 * 
	 * You are given a target value to search. If found in the array return its
	 * index, otherwise return -1.
	 * 
	 * You may assume no duplicate exists in the array.
	 */

	/*
	 * use two times binary search, first time to find pivot, second time to
	 * search value.
	 */

	public int search(int[] A, int target) {
		int pivot = 0;
		int start = 0, end = A.length - 1;
		while (start <= end) {
			int mid = (start + end) / 2;
			if (start == end) {
				pivot = start;
				break;
			}
			if (A[mid] > A[end]) {
				start = mid + 1;
			} else {
				end = mid;
			}
		}

		if (target == A[pivot]) {
			return pivot;
		} else if (target > A[pivot] && target <= A[A.length - 1]) {
			start = pivot;
			end = A.length - 1;
		} else {
			start = 0;
			end = pivot - 1;
		}
		while (start <= end) {
			int mid = (start + end) / 2;
			if (target == A[mid]) {
				return mid;
			} else if (target < A[mid]) {
				end = mid - 1;
			} else {
				start = mid + 1;
			}
		}
		return -1;
	}

	/**
	 * $(Search in Rotated Sorted Array II)
	 * 
	 * Follow up for "Search in Rotated Sorted Array": What if duplicates are
	 * allowed?
	 * 
	 * Would this affect the run-time complexity? How and why?
	 * 
	 * Write a function to determine if a given target is in the array.
	 */

	/*
	 * More generic solution, can be used in Search in Rotated Sorted Array I.
	 */
	public boolean search2(int[] A, int target) {
		int init = 0;
		while (init < A.length) {
			int start = init, end = A.length - 1;
			while (start <= end) {
				int mid = (start + end) / 2;
				if (target == A[mid]) {
					return true;
				}
				if (A[mid] > A[start]) {
					if (target >= A[start] && target < A[mid]) {
						end = mid - 1;
					} else {
						start = mid + 1;
					}
				} else if (A[mid] < A[start]) {
					if (target > A[mid] && target <= A[end]) {
						start = mid + 1;
					} else {
						end = mid - 1;
					}
				} else {
					init++;
					break;
				}
			}
		}
		return false;
	}

	/**
	 * $(Two Sum)
	 * 
	 * Given an array of integers, find two numbers such that they add up to a
	 * specific target number.
	 * 
	 * The function twoSum should return indices of the two numbers such that
	 * they add up to the target, where index1 must be less than index2. Please
	 * note that your returned answers (both index1 and index2) are not
	 * zero-based.
	 * 
	 * You may assume that each input would have exactly one solution.
	 * 
	 * Input: numbers={2, 7, 11, 15}, target=9 Output: index1=1, index2=2
	 */

	/*
	 * O(n^2), no extra space
	 */
	public int[] twoSumBruteForce(int[] numbers, int target) {
		int len = numbers.length;
		boolean stop = false;
		int[] result = new int[2];
		for (int i = 0; i < len - 1 && !stop; i++) {
			for (int j = i + 1; j < len; j++) {
				if (numbers[i] + numbers[j] == target) {
					stop = true;
					result[0] = i + 1;
					result[1] = j + 1;
					break;
				}
			}
		}
		return result;
	}

	/*
	 * O(n) time, O(n) space
	 */
	public int[] twoSum(int[] numbers, int target) {
		int[] result = new int[2];
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < numbers.length; i++) {
			map.put(numbers[i], i);
		}
		for (int i = 0; i < numbers.length; i++) {
			int diff = target - numbers[i];
			if (map.containsKey(diff)) {
				result[0] = i + 1;
				result[1] = map.get(diff) + 1;
				break;
			}
		}
		return result;
	}

	/**
	 * $(3Sum Closest)
	 * 
	 * Given an array S of n integers, find three integers in S such that the
	 * sum is closest to a given number, target. Return the sum of the three
	 * integers. You may assume that each input would have exactly one solution.
	 * 
	 * For example, given array S = {-1 2 1 -4}, and target = 1.
	 * 
	 * The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
	 */

	/*
	 * Make sure the array is sorted!
	 */
	public int threeSumClosest(int[] num, int target) {
		Arrays.sort(num);
		int len = num.length;
		int result = num[0] + num[1] + num[len - 1];
		for (int i = 0; i < len - 2; i++) {
			int start = i + 1, end = len - 1;
			while (start < end) {
				int sum = num[i] + num[start] + num[end];
				if (sum == target) {
					return target;
				}
				if (Math.abs(target - sum) < Math.abs(target - result)) {
					result = sum;
				}
				if (sum > target) {
					end--;
				} else {
					start++;
				}
			}
			while (i < len - 2 && num[i] == num[i + 1]) {
				i++;
			}
		}
		return result;
	}

	/**
	 * $(3Sum)
	 * 
	 * Given an array S of n integers, are there elements a, b, c in S such that
	 * a + b + c = 0? Find all unique triplets in the array which gives the sum
	 * of zero.
	 * 
	 * Note: Elements in a triplet (a,b,c) must be in non-descending order. (ie,
	 * a ≤ b ≤ c) The solution set must not contain duplicate triplets. For
	 * example, given array S = {-1 0 1 2 -1 -4},
	 * 
	 * A solution set is: (-1, 0, 1) (-1, -1, 2)
	 */

	public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		Arrays.sort(num);
		int len = num.length;
		for (int i = 0; i < len - 2; i++) {
			int start = i + 1, end = len - 1;
			while (start < end) {
				int sum = num[i] + num[start] + num[end];
				if (sum == 0) {
					ArrayList<Integer> triplet = new ArrayList<Integer>();
					triplet.add(num[i]);
					triplet.add(num[start]);
					triplet.add(num[end]);
					result.add(triplet);
					while (start < end && num[start] == num[start + 1]) {
						start++;
					}
					start++;
					while (end > start && num[end] == num[end - 1]) {
						end--;
					}
					end--;
				} else if (sum > 0) {
					end--;
				} else {
					start++;
				}
			}
			while (i < len - 2 && num[i] == num[i + 1]) {
				i++;
			}
		}

		return result;
	}

	/**
	 * $(4Sum)
	 * 
	 * Given an array S of n integers, are there elements a, b, c, and d in S
	 * such that a + b + c + d = target? Find all unique quadruplets in the
	 * array which gives the sum of target.
	 * 
	 * Note: Elements in a quadruplet (a,b,c,d) must be in non-descending order.
	 * (ie, a ≤ b ≤ c ≤ d) The solution set must not contain duplicate
	 * quadruplets. For example, given array S = {1 0 -1 0 -2 2}, and target =
	 * 0.
	 * 
	 * A solution set is: (-1, 0, 0, 1) (-2, -1, 1, 2) (-2, 0, 0, 2)
	 */

	/*
	 * precompute a map of two sum (collect pair from back to start), then use
	 * two layer for loop to fix the first two number (from start to end) then
	 * get the other two number form the map. Skip duplicated elements in both
	 * steps.
	 */
	public ArrayList<ArrayList<Integer>> fourSum(int[] num, int target) {
		if (num == null || num.length < 4) {
			return new ArrayList<ArrayList<Integer>>();
		}

		Arrays.sort(num);
		HashMap<Integer, ArrayList<ArrayList<Integer>>> map = new HashMap<Integer, ArrayList<ArrayList<Integer>>>();
		int len = num.length;
		for (int i = len - 1; i > 0; i--) {
			for (int j = i - 1; j >= 0;) {
				int sum = num[i] + num[j];
				ArrayList<Integer> pair = new ArrayList<Integer>(4);
				pair.add(j);
				pair.add(i);
				pair.add(num[j]);
				pair.add(num[i]);
				if (map.containsKey(sum)) {
					map.get(sum).add(pair);
				} else {
					ArrayList<ArrayList<Integer>> pairList = new ArrayList<ArrayList<Integer>>();
					pairList.add(pair);
					map.put(sum, pairList);
				}
				for (j = j - 1; j >= 0 && num[j] == num[j + 1];) {
					j--;
				}
			}
			while (i > 0 && num[i] == num[i - 1]) {
				i--;
			}
		}

		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		for (int i = 0; i < len - 1; i++) {
			for (int j = i + 1; j < len;) {
				int sum = num[i] + num[j];
				int key = target - sum;
				ArrayList<ArrayList<Integer>> pairList = map.get(key);
				if (pairList != null) {
					for (ArrayList<Integer> pair : pairList) {
						if (pair.get(0) > j) {
							ArrayList<Integer> quadruplet = new ArrayList<Integer>(
									4);
							quadruplet.add(num[i]);
							quadruplet.add(num[j]);
							quadruplet.add(pair.get(2));
							quadruplet.add(pair.get(3));
							result.add(quadruplet);
						}
					}
				}
				do {
					j++;
				} while (j < len && num[j] == num[j - 1]);
			}
			while (i < len - 1 && num[i] == num[i + 1]) {
				i++;
			}
		}

		return result;
	}

	/**
	 * $(Minimum Depth of Binary Tree)
	 * 
	 * Given a binary tree, find its minimum depth.
	 * 
	 * The minimum depth is the number of nodes along the shortest path from the
	 * root node down to the nearest leaf node.
	 */

	public int minDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		if (root.left == null && root.right == null) {
			return 1;
		}
		if (root.left != null && root.right != null) {
			return 1 + Math.min(minDepth(root.left), minDepth(root.right));
		} else if (root.left != null) {
			return 1 + minDepth(root.left);
		} else {
			return 1 + minDepth(root.right);
		}
	}

	/**
	 * $(Valid Parentheses)
	 * 
	 * Given a string containing just the characters '(', ')', '{', '}', '[' and
	 * ']', determine if the input string is valid.
	 * 
	 * The brackets must close in the correct order, "()" and "()[]{}" are all
	 * valid but "(]" and "([)]" are not.
	 */

	public boolean isValid(String s) {
		if (s == null || "".equals(s)) {
			return true;
		}
		int len = s.length();
		if (len % 2 == 1) {
			return false;
		}
		Stack<Character> stack = new Stack<Character>();
		HashMap<Character, Character> pair = new HashMap<Character, Character>();
		pair.put('(', ')');
		pair.put('[', ']');
		pair.put('{', '}');
		for (int i = 0; i < len; i++) {
			char c = s.charAt(i);
			if (pair.containsKey(c)) {
				stack.push(c);
			} else {
				if (stack.isEmpty()) {
					return false;
				} else {
					char ch = stack.pop();
					if (pair.get(ch) != c) {
						return false;
					}
				}
			}
		}
		if (!stack.isEmpty()) {
			return false;
		}
		return true;
	}

	/**
	 * $(Sum Root to Leaf Numbers)
	 * 
	 * Given a binary tree containing digits from 0-9 only, each root-to-leaf
	 * path could represent a number.
	 * 
	 * An example is the root-to-leaf path 1->2->3 which represents the number
	 * 123.
	 * 
	 * Find the total sum of all root-to-leaf numbers.
	 * 
	 * For example, {1, 2, 3}
	 * 
	 * The root-to-leaf path 1->2 represents the number 12. The root-to-leaf
	 * path 1->3 represents the number 13.
	 * 
	 * Return the sum = 12 + 13 = 25.
	 */

	public int sumNumbers(TreeNode root) {
		if (root == null) {
			return 0;
		}
		ArrayList<Integer> paths = new ArrayList<Integer>();
		getPaths(paths, root, 0);
		int sum = 0;
		for (Integer num : paths) {
			sum += num;
		}
		return sum;
	}

	private void getPaths(ArrayList<Integer> paths, TreeNode node, int base) {
		int curr = base * 10 + node.val;
		if (node.left == null && node.right == null) {
			paths.add(curr);
			return;
		}
		if (node.left != null) {
			getPaths(paths, node.left, curr);
		}
		if (node.right != null) {
			getPaths(paths, node.right, curr);
		}
	}

	/**
	 * $(Trapping Rain Water)
	 * 
	 * Given n non-negative integers representing an elevation map where the
	 * width of each bar is 1, compute how much water it is able to trap after
	 * raining.
	 * 
	 * For example, Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
	 */

	/*
	 * O(n^2), space O(n), not a good solution. Find all the peaks by comparing
	 * with left and right element, store indices of them into an array, then
	 * use two pointers point to the first and last peak, set water level to the
	 * height of the lower peak, then fill water between two ends, then move the
	 * lower peak toward the other one until it meets a higher peak, fill the
	 * water higher than last water level, and so forth.
	 */

	public int trapBadSolution(int[] A) {
		if (A == null || A.length < 3) {
			return 0;
		}
		int len = A.length;
		ArrayList<Integer> peaks = new ArrayList<Integer>();
		if (A[0] > A[1]) {
			peaks.add(0);
		}
		for (int i = 1; i < len - 1; i++) {
			if (A[i] > A[i - 1] && A[i] >= A[i + 1] || A[i] >= A[i - 1]
					&& A[i] > A[i + 1]) {
				peaks.add(i);
			}
		}
		if (A[len - 1] > A[len - 2]) {
			peaks.add(len - 1);
		}
		int size = peaks.size();
		if (size < 2) {
			return 0;
		}
		int currlv = 0, lastlv = 0, sum = 0;
		boolean mvright = true;
		int l = 0, r = size - 1;
		for (int start = peaks.get(l), end = peaks.get(r); start < end;) {
			if (A[start] > A[end]) {
				currlv = A[end];
				mvright = false;
			} else {
				currlv = A[start];
				mvright = true;
			}
			for (int i = start + 1; i < end; i++) {
				if (A[i] < currlv) {
					sum += currlv - Math.max(A[i], lastlv);
				}
			}
			lastlv = currlv;
			if (mvright) {
				int ptr = l + 1;
				while (ptr < r && A[peaks.get(ptr)] <= A[peaks.get(l)]) {
					ptr++;
				}
				start = peaks.get(ptr);
				l = ptr;
			} else {
				int ptr = r - 1;
				while (ptr > l && A[peaks.get(ptr)] <= A[peaks.get(r)]) {
					ptr--;
				}
				end = peaks.get(ptr);
				r = ptr;
			}
		}
		return sum;
	}

	/*
	 * O(n) time, no extra space. This solution is based on an observation that
	 * we can always use the height of the highest peak so far as water level
	 * unless there is no higher peak in the future. Thus we can scan the whole
	 * array to find the highest peak, then scan two times to fill water, one
	 * time from start to peak, the other time from end to peak, we use the same
	 * strategy to fill water: if current height is higher than current water
	 * level, update water level, if lower, fill water. Water level is actually
	 * the max value so far.
	 */

	public int trap(int[] A) {
		if (A == null || A.length < 3) {
			return 0;
		}
		int len = A.length, peak = 0;
		for (int i = 1; i < len; i++) {
			if (A[i] > A[peak]) {
				peak = i;
			}
		}
		int lv = 0, sum = 0;
		for (int i = 0; i < peak; i++) {
			if (A[i] < lv) {
				sum += lv - A[i];
			} else if (A[i] > lv) {
				lv = A[i];
			}
		}
		lv = 0;
		for (int i = len - 1; i > peak; i--) {
			if (A[i] < lv) {
				sum += lv - A[i];
			} else if (A[i] > lv) {
				lv = A[i];
			}
		}
		return sum;
	}

	/**
	 * $(Search for a Range)
	 * 
	 * Given a sorted array of integers, find the starting and ending position
	 * of a given target value.
	 * 
	 * Your algorithm's runtime complexity must be in the order of O(log n).
	 * 
	 * If the target is not found in the array, return [-1, -1].
	 * 
	 * For example, Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4].
	 */

	/*
	 * 1. use binary search to find the target, the l and r at the time it is
	 * found is the enclosing range. 2. use binary search to find the start and
	 * the end of the range, the start element should be either l or an element
	 * whose previous element is not equal to it, and it is the similar for the
	 * end element.
	 */
	public int[] searchRange(int[] A, int target) {
		if (A == null || A.length == 0 || target < A[0]
				|| target > A[A.length - 1]) {
			return new int[] { -1, -1 };
		}
		int len = A.length;
		int l = 0, r = len - 1;
		while (l <= r) {
			int mid = (l + r) / 2;
			if (A[mid] == target) {
				break;
			} else if (A[mid] > target) {
				r = mid - 1;
			} else {
				l = mid + 1;
			}
		}
		if (l > r) {
			return new int[] { -1, -1 };
		} else if (l == r) {
			return new int[] { l, r };
		} else {
			int anchor = (l + r) / 2;
			int start = l, end = anchor;
			int[] pair = new int[2];
			while (start <= end) {
				int mid = (start + end) / 2;
				if (A[mid] == target) {
					if (mid == l) {
						pair[0] = mid;
						break;
					} else {
						if (A[mid - 1] == A[mid]) {
							end = mid - 1;
						} else {
							pair[0] = mid;
							break;
						}
					}
				} else {
					start = mid + 1;
				}
			}
			start = anchor;
			end = r;
			while (start <= end) {
				int mid = (start + end) / 2;
				if (A[mid] == target) {
					if (mid == r) {
						pair[1] = mid;
						break;
					} else {
						if (A[mid + 1] == A[mid]) {
							start = mid + 1;
						} else {
							pair[1] = mid;
							break;
						}
					}
				} else {
					end = mid - 1;
				}
			}
			return pair;
		}
	}

	/**
	 * $(Insertion Sort List)
	 * 
	 * Sort a linked list using insertion sort.
	 */

	public ListNode insertionSortList(ListNode head) {
		if (head != null && head.next != null) {
			ListNode bp = head, fp = head.next;
			while (fp != null) {
				if (fp.val >= bp.val) {
					bp = fp;
					fp = fp.next;
				} else {
					ListNode curr = fp;
					fp = fp.next;
					bp.next = fp;
					if (curr.val < head.val) {
						curr.next = head;
						head = curr;
					} else {
						ListNode fptr = head.next, bptr = head;
						while (true) {
							if (curr.val < fptr.val) {
								curr.next = fptr;
								bptr.next = curr;
								break;
							} else {
								bptr = fptr;
								fptr = fptr.next;
							}
						}
					}
				}
			}

		}

		return head;
	}

	/**
	 * $(Longest Common Prefix)
	 * 
	 * Write a function to find the longest common prefix string amongst an
	 * array of strings.
	 */

	/*
	 * O(n^2)
	 */
	public String longestCommonPrefix(String[] strs) {
		String lcp = "";
		if (strs == null || strs.length == 0) {
			return lcp;
		}
		if (strs.length == 1) {
			return strs[0];
		}
		int maxlen = strs[0].length(), len = strs.length;
		for (int i = 1; i < len; i++) {
			if (strs[i].length() < maxlen) {
				maxlen = strs[i].length();
			}
		}
		boolean stop = false;
		for (int i = 0; i < maxlen && !stop; i++) {
			for (int j = 1; j < len; j++) {
				if (strs[j].charAt(i) != strs[j - 1].charAt(i)) {
					stop = true;
					break;
				}
			}
			if (!stop) {
				lcp += strs[0].charAt(i);
			}
		}
		return lcp;
	}

	/**
	 * $(Subsets II)
	 * 
	 * Given a collection of integers that might contain duplicates, S, return
	 * all possible subsets.
	 * 
	 * Note: Elements in a subset must be in non-descending order. The solution
	 * set must not contain duplicate subsets. For example, If S = [1,2,2], a
	 * solution is:
	 * 
	 * [ [2], [1], [1,2,2], [2,2], [1,2], [] ]
	 */

	public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null) {
			return result;
		}

		Arrays.sort(num);
		boolean[] visited = new boolean[num.length];
		ArrayList<Integer> combination = new ArrayList<Integer>();
		subsetsWithDupDFS(result, combination, num, visited, 0);

		return result;
	}

	private void subsetsWithDupDFS(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> combination, int[] num, boolean[] visited,
			int pos) {
		if (pos == visited.length) {
			result.add(new ArrayList<Integer>(combination));
			return;
		}

		if (!visited[pos]
				&& (pos == 0 || num[pos] != num[pos - 1] || visited[pos - 1])) {
			combination.add(num[pos]);
			visited[pos] = true;
			subsetsWithDupDFS(result, combination, num, visited, pos + 1);
			visited[pos] = false;
			combination.remove(combination.size() - 1);
		}
		subsetsWithDupDFS(result, combination, num, visited, pos + 1);
	}

	/**
	 * $(Jump Game)
	 * 
	 * Given an array of non-negative integers, you are initially positioned at
	 * the first index of the array.
	 * 
	 * Each element in the array represents your maximum jump length at that
	 * position.
	 * 
	 * Determine if you are able to reach the last index.
	 * 
	 * For example: A = [2,3,1,1,4], return true.
	 * 
	 * A = [3,2,1,0,4], return false.
	 */

	/*
	 * greedy solution : use a pointer to scan the array to extend the maximum
	 * reachable border until border pass the target or the pointer reaches the
	 * border. O(n)
	 */
	public boolean canJump(int[] A) {
		if (A == null || A.length == 0) {
			return false;
		}

		int len = A.length, start = 0, end = A[0];
		while (true) {
			if (start < len && start <= end) {
				end = Math.max(end, start + A[start]);
			} else {
				return false;
			}
			if (end >= len - 1) {
				return true;
			}
			start++;
		}
	}

	/*
	 * sequence DP, dp[i] denotes whether i can be reached from 0. To get dp[i],
	 * we try to find a position j, 0 <= j < i, that is reachable and can jump
	 * to i directly. O(n^2)
	 */
	public boolean canJumpDP(int[] A) {
		if (A == null || A.length == 0) {
			return false;
		}

		int len = A.length;
		boolean[] dp = new boolean[len];
		dp[0] = true;
		for (int i = 1; i < len; i++) {
			for (int j = 0; j < i; j++) {
				if (dp[j] && j + A[j] >= i) {
					dp[i] = true;
					break;
				}
			}
		}

		return dp[len - 1];
	}

	/**
	 * $(Path Sum II)
	 * 
	 * Given a binary tree and a sum, find all root-to-leaf paths where each
	 * path's sum equals the given sum.
	 * 
	 */

	/*
	 * Attention: shared reference in recursion is bug-prone.
	 */
	public ArrayList<ArrayList<Integer>> pathSum(TreeNode root, int sum) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (root != null) {
			ArrayList<Integer> path = new ArrayList<Integer>();
			getPathSum(result, path, root, sum);
		}
		return result;
	}

	private void getPathSum(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> path, TreeNode root, int sum) {
		if (root.left == null && root.right == null) {
			if (root.val == sum) {
				ArrayList<Integer> newPath = new ArrayList<Integer>();
				for (Integer n : path) {
					newPath.add(n);
				}
				newPath.add(root.val);
				result.add(newPath);
			}
			return;
		}
		if (root.left != null) {
			path.add(root.val);
			getPathSum(result, path, root.left, sum - root.val);
			path.remove(path.size() - 1);
		}
		if (root.right != null) {
			path.add(root.val);
			getPathSum(result, path, root.right, sum - root.val);
			path.remove(path.size() - 1);
		}
	}

	/**
	 * $(Length of Last Word)
	 * 
	 * Given a string s consists of upper/lower-case alphabets and empty space
	 * characters ' ', return the length of last word in the string.
	 * 
	 * If the last word does not exist, return 0.
	 * 
	 * Note: A word is defined as a character sequence consists of non-space
	 * characters only.
	 * 
	 * For example, Given s = "Hello World", return 5.
	 */

	public int lengthOfLastWord(String s) {
		if (s == null || s.length() == 0) {
			return 0;
		}
		int count = 0, ptr = s.length() - 1;
		while (ptr >= 0 && s.charAt(ptr) != ' ') {
			count++;
			ptr--;
		}
		return count;
	}

	/**
	 * $(Valid Sudoku)
	 * 
	 * Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.
	 * 
	 * The Sudoku board could be partially filled, where empty cells are filled
	 * with the character '.'.
	 */

	public boolean isValidSudoku(char[][] board) {
		if (board == null || board[0] == null || board.length != 9
				|| board[0].length != 9) {
			return false;
		}

		boolean[][] rowMap = new boolean[9][9], colMap = new boolean[9][9], gridMap = new boolean[9][9];
		for (int row = 0; row < 9; row++) {
			for (int col = 0; col < 9; col++) {
				if (board[row][col] != '.') {
					int num = board[row][col] - '0' - 1;
					if (rowMap[row][num] || colMap[col][num]
							|| gridMap[row / 3 * 3 + col / 3][num]) {
						return false;
					} else {
						rowMap[row][num] = true;
						colMap[col][num] = true;
						gridMap[row / 3 * 3 + col / 3][num] = true;
					}
				}
			}
		}

		return true;
	}

	/**
	 * $(Sudoku Solver)
	 * 
	 * Write a program to solve a Sudoku puzzle by filling the empty cells.
	 * 
	 * Empty cells are indicated by the character '.'.
	 * 
	 * You may assume that there will be only one unique solution.
	 */

	public void solveSudoku(char[][] board) {
		if (board == null || board[0] == null || board.length != 9
				|| board[0].length != 9) {
			return;
		}

		boolean[][] rowMap = new boolean[9][9], colMap = new boolean[9][9], gridMap = new boolean[9][9];
		getMaps(board, rowMap, colMap, gridMap);

		solveSudokuDFS(board, rowMap, colMap, gridMap, 0);
	}

	private void getMaps(char[][] board, boolean[][] rowMap,
			boolean[][] colMap, boolean[][] gridMap) {
		for (int row = 0; row < 9; row++) {
			for (int col = 0; col < 9; col++) {
				if (board[row][col] != '.') {
					int num = board[row][col] - '0' - 1;
					rowMap[row][num] = true;
					colMap[col][num] = true;
					gridMap[row / 3 * 3 + col / 3][num] = true;
				}
			}
		}
	}

	private boolean solveSudokuDFS(char[][] board, boolean[][] rowMap,
			boolean[][] colMap, boolean[][] gridMap, int index) {
		if (index == 81) {
			return true;
		}

		int currRow = index / 9, currCol = index % 9;
		char val = board[currRow][currCol];
		if (val == '.') {
			for (int i = 0; i < 9; i++) {
				if (!rowMap[currRow][i] && !colMap[currCol][i]
						&& !gridMap[currRow / 3 * 3 + currCol / 3][i]) {
					rowMap[currRow][i] = true;
					colMap[currCol][i] = true;
					gridMap[currRow / 3 * 3 + currCol / 3][i] = true;
					board[currRow][currCol] = (char) ('0' + i + 1);
					if (solveSudokuDFS(board, rowMap, colMap, gridMap,
							index + 1)) {
						return true;
					}
					rowMap[currRow][i] = false;
					colMap[currCol][i] = false;
					gridMap[currRow / 3 * 3 + currCol / 3][i] = false;
					board[currRow][currCol] = val;
				}
			}
			return false;
		} else {
			return solveSudokuDFS(board, rowMap, colMap, gridMap, index + 1);
		}
	}

	/**
	 * $(Unique Paths II)
	 * 
	 * Follow up for "Unique Paths":
	 * 
	 * Now consider if some obstacles are added to the grids. How many unique
	 * paths would there be?
	 * 
	 * An obstacle and empty space is marked as 1 and 0 respectively in the
	 * grid.
	 * 
	 * For example, There is one obstacle in the middle of a 3x3 grid as
	 * illustrated below.
	 * 
	 * [ [0,0,0], [0,1,0], [0,0,0] ] The total number of unique paths is 2.
	 * 
	 * Note: m and n will be at most 100.
	 */

	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		if (obstacleGrid == null || obstacleGrid.length == 0
				|| obstacleGrid[0] == null || obstacleGrid[0].length == 0) {
			return 0;
		}

		int height = obstacleGrid.length, width = obstacleGrid[0].length;
		int[][] dp = new int[height][width];
		if (obstacleGrid[height - 1][width - 1] == 1) {
			dp[height - 1][width - 1] = 0;
		} else {
			dp[height - 1][width - 1] = 1;
		}
		for (int col = width - 2; col >= 0; col--) {
			dp[height - 1][col] = (obstacleGrid[height - 1][col] == 1 ? 0
					: dp[height - 1][col + 1]);
		}
		for (int row = height - 2; row >= 0; row--) {
			dp[row][width - 1] = (obstacleGrid[row][width - 1] == 1 ? 0
					: dp[row + 1][width - 1]);
		}

		for (int row = height - 2; row >= 0; row--) {
			for (int col = width - 2; col >= 0; col--) {
				if (obstacleGrid[row][col] == 1) {
					dp[row][col] = 0;
				} else {
					dp[row][col] = dp[row][col + 1] + dp[row + 1][col];
				}
			}
		}

		return dp[0][0];
	}

	/**
	 * $(Count and Say)
	 * 
	 * The count-and-say sequence is the sequence of integers beginning as
	 * follows: 1, 11, 21, 1211, 111221, ...
	 * 
	 * 1 is read off as "one 1" or 11. 11 is read off as "two 1s" or 21. 21 is
	 * read off as "one 2, then one 1" or 1211. Given an integer n, generate the
	 * nth sequence.
	 * 
	 * Note: The sequence of integers will be represented as a string.
	 */

	public String countAndSay(int n) {
		String result = "1";
		for (int i = 1; i < n; i++) {
			result = getRead(result);
		}
		return result;
	}

	private String getRead(String str) {
		int len = str.length();
		String result = "";
		for (int i = 0; i < len; i++) {
			char curr = str.charAt(i);
			int count = 1;
			while (i < len - 1 && str.charAt(i) == str.charAt(i + 1)) {
				i++;
				count++;
			}
			result += String.valueOf(count) + String.valueOf(curr);
		}
		return result;
	}

	/**
	 * $(Evaluate Reverse Polish Notation)
	 * 
	 * Evaluate the value of an arithmetic expression in Reverse Polish
	 * Notation.
	 * 
	 * Valid operators are +, -, *, /. Each operand may be an integer or another
	 * expression. Some examples:
	 * 
	 * ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
	 * 
	 * ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
	 */

	public int evalRPN(String[] tokens) {
		Stack<Integer> s = new Stack<Integer>();
		int len = tokens.length;
		for (int i = 0; i < len; i++) {
			String str = tokens[i];
			if (str.matches("-?\\d+")) {
				s.push(Integer.parseInt(str));
			} else {
				processOperand(str, s);
			}
		}
		return s.pop();
	}

	private void processOperand(String str, Stack<Integer> s) {
		int op2 = s.pop();
		int op1 = s.pop();
		char c = str.charAt(0);
		if (c == '+') {
			s.push(op1 + op2);
		} else if (c == '-') {
			s.push(op1 - op2);
		} else if (c == '*') {
			s.push(op1 * op2);
		} else {
			s.push(op1 / op2);
		}
	}

	/**
	 * $(Add Two Numbers)
	 * 
	 * You are given two linked lists representing two non-negative numbers. The
	 * digits are stored in reverse order and each of their nodes contain a
	 * single digit. Add the two numbers and return it as a linked list.
	 * 
	 * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
	 * 
	 * Output: 7 -> 0 -> 8
	 */

	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		if (l1 == null) {
			return l2;
		} else if (l2 == null) {
			return l1;
		}

		ListNode dummy = new ListNode(0), ptr = dummy;
		int carry = 0;
		while (l1 != null && l2 != null) {
			int sum = l1.val + l2.val + carry;
			ptr.next = new ListNode(sum % 10);
			ptr = ptr.next;
			carry = sum / 10;
			l1 = l1.next;
			l2 = l2.next;
		}

		if (l1 == null) {
			l1 = l2;
		}

		while (l1 != null) {
			int sum = l1.val + carry;
			ptr.next = new ListNode(sum % 10);
			ptr = ptr.next;
			carry = sum / 10;
			l1 = l1.next;
		}

		if (carry == 1) {
			ptr.next = new ListNode(1);
		}

		return dummy.next;
	}

	/**
	 * $(Triangle)
	 * 
	 * Given a triangle, find the minimum path sum from top to bottom. Each step
	 * you may move to adjacent numbers on the row below.
	 * 
	 * For example, given the following triangle
	 * 
	 * [ [2], [3,4], [6,5,7], [4,1,8,3] ]
	 * 
	 * The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
	 * 
	 * Note: Bonus point if you are able to do this using only O(n) extra space,
	 * where n is the total number of rows in the triangle.
	 */

	public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
		if (triangle == null) {
			return 0;
		}

		ArrayList<Integer> last = triangle.get(triangle.size() - 1);
		for (int i = triangle.size() - 2; i >= 0; i--) {
			ArrayList<Integer> newRow = new ArrayList<Integer>();
			ArrayList<Integer> curr = triangle.get(i);
			for (int j = 0; j < curr.size(); j++) {
				newRow.add(curr.get(j) + Math.min(last.get(j), last.get(j + 1)));
			}
			last = newRow;
		}

		return last.get(0);
	}

	/**
	 * $(Gas Station)
	 * 
	 * There are N gas stations along a circular route, where the amount of gas
	 * at station i is gas[i].
	 * 
	 * You have a car with an unlimited gas tank and it costs cost[i] of gas to
	 * travel from station i to its next station (i+1). You begin the journey
	 * with an empty tank at one of the gas stations.
	 * 
	 * Return the starting gas station's index if you can travel around the
	 * circuit once, otherwise return -1.
	 * 
	 * Note: The solution is guaranteed to be unique.
	 */

	/*
	 * Brute force solution would be O(n^2). This is a greedy solution, use a
	 * pointer start to indicate the start of a journey, and end to indicate the
	 * end of a journey, and an int due to record the accumulated difference of
	 * gas and cost, if due >= 0, it means the path is through, end will move by
	 * one, otherwise the path is not through and we have to move start to the
	 * previous station. Loop it until the start meets the end, if at this point
	 * due is negative, then there is not solution, otherwise the point itself
	 * is the solution.
	 */
	public int canCompleteCircuit(int[] gas, int[] cost) {
		int len = gas.length;
		int start = 0, end = (start + 1) % len, curr = gas[0] - cost[0];
		while (start != end) {
			if (curr >= 0) {
				curr += gas[end] - cost[end];
				end = (end + 1) % len;
			} else {
				start = (start + len - 1) % len;
				curr += gas[start] - cost[start];
			}
		}
		if (curr >= 0) {
			return start;
		} else {
			return -1;
		}
	}

	/**
	 * $(Candy)
	 * 
	 * There are N children standing in a line. Each child is assigned a rating
	 * value.
	 * 
	 * You are giving candies to these children subjected to the following
	 * requirements:
	 * 
	 * Each child must have at least one candy. Children with a higher rating
	 * get more candies than their neighbors. What is the minimum candies you
	 * must give?
	 */

	/*
	 * Scan two times, first time from start to end, if current kid has higher
	 * rating then previous kid, then give him one more then previous one. Then
	 * do the same thing from end to start. Pay attention to the case when the
	 * peak is higher in front side.
	 */
	public int candy(int[] ratings) {
		if (ratings == null) {
			return 0;
		}

		int len = ratings.length;
		int[] candies = new int[len];
		for (int i = 0; i < len; i++) {
			candies[i] = 1;
		}
		for (int i = 1; i < len; i++) {
			if (ratings[i] > ratings[i - 1]) {
				candies[i] = candies[i - 1] + 1;
			}
		}
		for (int i = len - 2; i >= 0; i--) {
			if (ratings[i] > ratings[i + 1]) {
				int v = candies[i + 1] + 1;
				if (candies[i] < v) {
					candies[i] = v;
				}
			}
		}
		int sum = 0;
		for (int i = 0; i < len; i++) {
			sum += candies[i];
		}
		return sum;
	}

	/**
	 * $(Longest Consecutive Sequence)
	 * 
	 * Given an unsorted array of integers, find the length of the longest
	 * consecutive elements sequence.
	 * 
	 * For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive
	 * elements sequence is [1, 2, 3, 4]. Return its length: 4.
	 * 
	 * Your algorithm should run in O(n) complexity.
	 */

	public int longestConsecutive(int[] num) {
		if (num == null) {
			return 0;
		}

		Set<Integer> set = new HashSet<Integer>();
		for (int n : num) {
			set.add(n);
		}

		int maxLen = 1;
		for (int n : num) {
			int left = n - 1, right = n + 1;
			set.remove(n);

			while (set.contains(left)) {
				set.remove(left);
				left--;
			}

			while (set.contains(right)) {
				set.remove(right);
				right++;
			}

			maxLen = Math.max(right - left - 1, maxLen);
		}

		return maxLen;
	}

	/**
	 * $(N-Queens)
	 * 
	 * The n-queens puzzle is the problem of placing n queens on an n×n
	 * chessboard such that no two queens attack each other. Given an integer n,
	 * return all distinct solutions to the n-queens puzzle.
	 * 
	 * Each solution contains a distinct board configuration of the n-queens'
	 * placement, where 'Q' and '.' both indicate a queen and an empty space
	 * respectively.
	 * 
	 * For example, There exist two distinct solutions to the 4-queens puzzle:
	 * 
	 * [ [".Q..", "...Q", "Q...", "..Q."],
	 * 
	 * ["..Q.", "Q...", "...Q", ".Q.."] ]
	 */

	/*
	 * simple implementation, recommended
	 */
	public ArrayList<String[]> solveNQueens(int n) {
		ArrayList<String[]> result = new ArrayList<String[]>();
		if (n <= 0) {
			return result;
		}

		int[] rows = new int[n];
		solveNQueensDFS(result, rows, 0);

		return result;
	}

	private void solveNQueensDFS(ArrayList<String[]> result, int[] rows,
			int currRow) {
		if (currRow == rows.length) {
			String[] board = new String[rows.length];
			for (int r = 0; r < board.length; r++) {
				char[] chs = new char[rows.length];
				Arrays.fill(chs, '.');
				chs[rows[r]] = 'Q';
				board[r] = new String(chs);
			}
			result.add(board);
			return;
		}

		for (int c = 0; c < rows.length; c++) {
			if (canPut(rows, currRow, c)) {
				rows[currRow] = c;
				solveNQueensDFS(result, rows, currRow + 1);
			}
		}
	}

	/*
	 * use bitmap to store queen position
	 */
	public ArrayList<String[]> solveNQueensBitMap(int n) {
		ArrayList<String[]> result = new ArrayList<String[]>();
		if (n == 1 || n > 3) {
			boolean[][] board = new boolean[n][n];
			int col = 0, slash = 0, bkslash = 0;
			solveNQueens(result, board, col, slash, bkslash, 0);
		}
		return result;
	}

	private void solveNQueens(ArrayList<String[]> result, boolean[][] board,
			int col, int slash, int bkslash, int row) {
		int len = board.length;
		if (row == len) {
			String[] strs = new String[len];
			for (int r = 0; r < len; r++) {
				strs[r] = "";
				for (int c = 0; c < len; c++) {
					if (board[r][c]) {
						strs[r] += "Q";
					} else {
						strs[r] += ".";
					}
				}
			}
			result.add(strs);
			return;
		}

		for (int c = 0; c < len; c++) {
			int mcol = 1 << c;
			int mslash = 1 << (row + c);
			int mbkslash = 1 << (row - c + len - 1);
			if ((col & mcol | slash & mslash | bkslash & mbkslash) == 0) {
				board[row][c] = true;
				col |= mcol;
				slash |= mslash;
				bkslash |= mbkslash;
				solveNQueens(result, board, col, slash, bkslash, row + 1);
				board[row][c] = false;
				col ^= mcol;
				slash ^= mslash;
				bkslash ^= mbkslash;
			}
		}
	}

	/**
	 * $(N-Queens II)
	 * 
	 * Follow up for N-Queens problem.
	 * 
	 * Now, instead outputting board configurations, return the total number of
	 * distinct solutions.
	 */

	/*
	 * simple implementation, recommended
	 */
	public int totalNQueens(int n) {
		if (n < 1) {
			return 0;
		}

		int[] result = new int[1];
		int[] rows = new int[n];
		totalNQueens(result, rows, 0);
		return result[0];
	}

	private void totalNQueens(int[] result, int[] rows, int row) {
		int n = rows.length;
		if (row == n) {
			result[0]++;
			return;
		}

		for (int col = 0; col < n; col++) {
			if (canPut(rows, row, col)) {
				rows[row] = col;
				totalNQueens(result, rows, row + 1);
				rows[row] = 0;
			}
		}
	}

	private boolean canPut(int[] rows, int currRow, int currCol) {
		for (int r = 0; r < currRow; r++) {
			if (rows[r] == currCol
					|| currRow - r == Math.abs(currCol - rows[r])) {
				return false;
			}
		}
		return true;
	}

	/*
	 * use bitmap to save board info
	 */
	public int totalNQueensBitMap(int n) {
		if (n == 1) {
			return 1;
		} else if (n < 4) {
			return 0;
		}

		int[] result = new int[(n + 1) / 2];
		int col = 0, slash = 0, bkslash = 0;
		for (int i = 0; i < result.length; i++) {
			int mcol = 1 << i;
			int mslash = 1 << i;
			int mbkslash = 1 << (0 - i + n - 1);
			col |= mcol;
			slash |= mslash;
			bkslash |= mbkslash;
			totalNQueens(n, i, result, col, slash, bkslash, 1);
			col ^= mcol;
			slash ^= mslash;
			bkslash ^= mbkslash;
		}

		int sum = 0;
		for (int i = 0; i < result.length; i++) {
			sum += result[i];
		}
		System.out.println(Arrays.toString(result));
		System.out.println(sum);
		if (n % 2 == 0) {
			return sum * 2;
		} else {
			return sum * 2 - result[result.length - 1];
		}
	}

	private void totalNQueens(int len, int i, int[] result, int col, int slash,
			int bkslash, int r) {
		if (r == len) {
			result[i]++;
			return;
		}

		for (int c = 0; c < len; c++) {
			int mcol = 1 << c;
			int mslash = 1 << (r + c);
			int mbkslash = 1 << (r - c + len - 1);
			if ((col & mcol | slash & mslash | bkslash & mbkslash) == 0) {
				col |= mcol;
				slash |= mslash;
				bkslash |= mbkslash;
				totalNQueens(len, i, result, col, slash, bkslash, r + 1);
				col ^= mcol;
				slash ^= mslash;
				bkslash ^= mbkslash;
			}
		}
	}

	/*
	 * bit operation solution
	 */

	public int totalNQueensBitOp(int n) {
		int upperlim = (1 << n) - 1;
		int[] result = new int[1];
		totalNQueensBitOp(result, upperlim, 0, 0, 0);
		return result[0];
	}

	private void totalNQueensBitOp(int[] result, int upperlim, int col,
			int slash, int bkslash) {
		if (col == upperlim) {
			result[0]++;
			return;
		}

		int pos = upperlim & ~(col | slash | bkslash);
		while (pos != 0) {
			int p = pos & -pos;
			pos -= p;
			totalNQueensBitOp(result, upperlim, col + p, (slash + p) << 1,
					(bkslash + p) >> 1);
		}
	}

	/**
	 * $(Flatten Binary Tree to Linked List)
	 * 
	 * Given a binary tree, flatten it to a linked list in-place. Each node's
	 * right child points to the next node of a pre-order traversal.
	 */

	public void flatten(TreeNode root) {
		if (root != null) {
			TreeNode left = root.left, right = root.right;
			root.left = null;
			flatten(left);
			root.right = left;
			TreeNode ptr = root;
			while (ptr.right != null) {
				ptr = ptr.right;
			}
			flatten(right);
			ptr.right = right;
		}
	}

	/**
	 * $(Maximal Rectangle)
	 * 
	 * Given a 2D binary matrix filled with 0's and 1's, find the largest
	 * rectangle containing all ones and return its area.
	 */

	/*
	 * Brute force O(n^4), calculate maximum area at each '1'.
	 */
	public int maximalRectangle(char[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null
				|| matrix[0].length == 0) {
			return 0;
		}
		int w = matrix[0].length;
		int h = matrix.length;
		int max = 0;
		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) {
				if (matrix[r][c] == '1') {
					int area = getMaxArea(matrix, r, c);
					if (area > max) {
						max = area;
					}
				}
			}
		}
		return max;
	}

	private int getMaxArea(char[][] matrix, int r, int c) {
		int w = matrix[0].length;
		int h = matrix.length;
		int depth = r;
		while (depth < h - 1 && matrix[depth + 1][c] == '1') {
			depth++;
		}
		ArrayList<Integer> arr = new ArrayList<Integer>(depth - r + 1);
		int border = w;
		for (int i = r; i <= depth; i++) {
			int width = c;
			while (width < border - 1 && matrix[i][width + 1] == '1') {
				width++;
			}
			arr.add(width - c + 1);
			if (width < border - 1) {
				border = width + 1;
			}
		}
		int maxArea = 0;
		for (int i = 0; i < arr.size(); i++) {
			int area = (i + 1) * arr.get(i);
			if (area > maxArea) {
				maxArea = area;
			}
		}
		return maxArea;
	}

	/*
	 * http://hi.baidu.com/mzry1992/item/030f9740e0475ef7dc0f6cba
	 */

	public int maximalRectangleHangingLine(char[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null
				|| matrix[0].length == 0) {
			return 0;
		}

		int w = matrix[0].length;
		int h = matrix.length;
		int[] ht = new int[w];
		int[] l = new int[w];
		int[] r = new int[w];
		Arrays.fill(r, w);
		int maxArea = 0;

		for (int row = 0; row < h; row++) {
			int left = 0, right = w;
			for (int col = 0; col < w; col++) {
				if (matrix[row][col] == '0') {
					left = col + 1;
					ht[col] = 0;
					l[col] = 0;
				} else {
					ht[col]++;
					l[col] = Math.max(l[col], left);
				}
			}
			for (int col = w - 1; col >= 0; col--) {
				if (matrix[row][col] == '0') {
					right = col;
					r[col] = w;
				} else {
					r[col] = Math.min(r[col], right);
					maxArea = Math.max(maxArea, (r[col] - l[col]) * ht[col]);
				}
			}
		}

		return maxArea;
	}

	/*
	 * Transfer the problem to solve "Largest Rectangle in Histogram" for each
	 * row. O(n * m)
	 */

	public int maximalRectangleHistogram(char[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null
				|| matrix[0].length == 0) {
			return 0;
		}

		int ht = matrix.length, wd = matrix[0].length, maxArea = 0;
		int[] heights = new int[wd];
		for (int row = 0; row < ht; row++) {
			for (int col = 0; col < wd; col++) {
				if (matrix[row][col] == '0') {
					heights[col] = 0;
				} else {
					heights[col]++;
				}
			}
			maxArea = Math.max(maxArea, getMaxArea(heights));
		}

		return maxArea;
	}

	private int getMaxArea(int[] heights) {
		int maxArea = 0, len = heights.length;
		Stack<Integer> pos = new Stack<Integer>();
		Stack<Integer> hgt = new Stack<Integer>();
		pos.push(0);
		hgt.push(heights[0]);
		for (int i = 1; i < len; i++) {
			if (heights[i] >= hgt.peek()) {
				pos.push(i);
				hgt.push(heights[i]);
			} else {
				int lastPos = pos.peek();
				while (!pos.isEmpty() && heights[i] < hgt.peek()) {
					lastPos = pos.peek();
					maxArea = Math.max(maxArea, (i - pos.pop()) * hgt.pop());
				}
				pos.push(lastPos);
				hgt.push(heights[i]);
			}
		}

		while (!pos.isEmpty()) {
			maxArea = Math.max(maxArea, (len - pos.pop()) * hgt.pop());
		}

		return maxArea;
	}

	/**
	 * $(Largest Rectangle in Histogram)
	 * 
	 * Given n non-negative integers representing the histogram's bar height
	 * where the width of each bar is 1, find the area of largest rectangle in
	 * the histogram.
	 * 
	 * For example, Given height = [2,1,5,6,2,3], return 10.
	 */

	public int largestRectangleArea(int[] height) {
		if (height == null || height.length == 0) {
			return 0;
		}

		Stack<Integer> posStack = new Stack<Integer>();
		Stack<Integer> hgtStack = new Stack<Integer>();
		int maxArea = 0, len = height.length;
		posStack.push(0);
		hgtStack.push(height[0]);
		for (int i = 1; i < len; i++) {
			if (height[i] >= hgtStack.peek()) {
				posStack.push(i);
				hgtStack.push(height[i]);
			} else {
				int lastPos = posStack.peek();
				while (!hgtStack.isEmpty() && hgtStack.peek() > height[i]) {
					lastPos = posStack.peek();
					maxArea = Math.max(maxArea,
							(i - posStack.pop()) * hgtStack.pop());
				}
				posStack.push(lastPos);
				hgtStack.push(height[i]);
			}
		}

		while (!posStack.isEmpty()) {
			maxArea = Math
					.max(maxArea, (len - posStack.pop()) * hgtStack.pop());
		}

		return maxArea;
	}

	/**
	 * $(Rotate List)
	 * 
	 * Given a list, rotate the list to the right by k places, where k is
	 * non-negative.
	 * 
	 * For example: Given 1->2->3->4->5->NULL and k = 2, return
	 * 4->5->1->2->3->NULL.
	 */

	public ListNode rotateRight(ListNode head, int n) {
		if (head == null) {
			return head;
		}
		int count = 1;
		ListNode tail = head;
		while (tail.next != null) {
			tail = tail.next;
			count++;
		}
		n %= count;
		if (count == 1 || n == 0) {
			return head;
		}
		ListNode ptr = head;
		for (int i = 1; i < count - n; i++) {
			ptr = ptr.next;
		}
		tail.next = head;
		head = ptr.next;
		ptr.next = null;
		return head;
	}

	/**
	 * $(Valid Palindrome)
	 * 
	 * Given a string, determine if it is a palindrome, considering only
	 * alphanumeric characters and ignoring cases.
	 * 
	 * For example, "A man, a plan, a canal: Panama" is a palindrome.
	 * "race a car" is not a palindrome.
	 * 
	 * Note: Have you consider that the string might be empty? This is a good
	 * question to ask during an interview.
	 * 
	 * For the purpose of this problem, we define empty string as valid
	 * palindrome.
	 */

	public boolean isPalindrome(String s) {
		if (s == null) {
			return false;
		}

		char[] chs = s.toCharArray();
		int back = 0, front = 0;
		while (front < chs.length) {
			if (Character.isLetter(chs[front])) {
				chs[back] = Character.toLowerCase(chs[front]);
				back++;
			} else if (Character.isDigit(chs[front])) {
				chs[back] = chs[front];
				back++;
			}
			front++;
		}

		int start = 0, end = back - 1;
		while (start < end) {
			if (chs[start] != chs[end]) {
				return false;
			}
			start++;
			end--;
		}
		return true;
	}

	/**
	 * $(Add Binary)
	 * 
	 * Given two binary strings, return their sum (also a binary string).
	 * 
	 * For example, a = "11" b = "1" Return "100".
	 */

	public String addBinary(String a, String b) {
		if (a == null || a.length() == 0) {
			return b;
		} else if (b == null || b.length() == 0) {
			return a;
		}

		String temp;
		if (b.length() > a.length()) {
			temp = a;
			a = b;
			b = temp;
		}

		int[] l1 = new int[a.length()];
		int len = l1.length;
		for (int i = 0; i < len; i++) {
			l1[i] = a.charAt(len - 1 - i) - '0';
		}

		int[] l2 = new int[b.length()];
		len = l2.length;
		for (int i = 0; i < len; i++) {
			l2[i] = b.charAt(len - 1 - i) - '0';
		}

		int carry = 0, i = 0;
		for (; i < l2.length; i++) {
			l1[i] = l1[i] + l2[i] + carry;
			carry = 0;
			if (l1[i] >= 2) {
				l1[i] -= 2;
				carry = 1;
			}
		}
		String head = "";
		for (; i < l1.length && carry == 1; i++) {
			l1[i] = l1[i] + carry;
			carry = 0;
			if (l1[i] >= 2) {
				l1[i] -= 2;
				carry = 1;
			}
		}
		if (carry == 1) {
			head = "1";
		}
		String result = head;
		len = l1.length;
		for (i = len - 1; i >= 0; i--) {
			result += String.valueOf(l1[i]);
		}
		return result;
	}

	/**
	 * $(Merge Intervals)
	 * 
	 * Given a collection of intervals, merge all overlapping intervals.
	 * 
	 * For example, Given [1,3],[2,6],[8,10],[15,18], return
	 * [1,6],[8,10],[15,18].
	 */

	class Interval {
		int start;
		int end;

		Interval() {
			start = 0;
			end = 0;
		}

		Interval(int s, int e) {
			start = s;
			end = e;
		}
	}

	public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
		if (intervals == null || intervals.size() < 2) {
			return intervals;
		}

		Collections.sort(intervals, new IntervalCmp());
		int curr = 0, currEnd = intervals.get(curr).end, index = 1;
		while (index < intervals.size()) {
			if (intervals.get(index).start <= currEnd) {
				currEnd = Math.max(currEnd, intervals.get(index).end);
				intervals.get(curr).end = currEnd;
				intervals.remove(index);
			} else {
				currEnd = intervals.get(index).end;
				curr = index;
				index++;
			}
		}

		return intervals;
	}

	public class IntervalCmp implements Comparator<Interval> {
		public int compare(Interval i1, Interval i2) {
			return i1.start - i2.start;
		}
	}

	/**
	 * $(Best Time to Buy and Sell Stock III)
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i.
	 * 
	 * Design an algorithm to find the maximum profit. You may complete at most
	 * two transactions.
	 * 
	 * Note: You may not engage in multiple transactions at the same time (ie,
	 * you must sell the stock before you buy again).
	 */

	/*
	 * Complicated, O(n^2)
	 */
	public int maxProfit3Complicated(int[] prices) {
		if (prices == null || prices.length < 2) {
			return 0;
		}
		int len = prices.length;
		if (len == 2) {
			if (prices[0] < prices[1]) {
				return prices[1] - prices[0];
			} else {
				return 0;
			}
		}
		// pnb : peak and bottom
		ArrayList<Integer> pnb = new ArrayList<Integer>();
		int profit = 0;
		int i = 0;
		while (i < len - 1) {
			while (i < len - 1 && prices[i] >= prices[i + 1]) {
				i++;
			}
			pnb.add(prices[i]);
			while (i < len - 1 && prices[i] < prices[i + 1]) {
				i++;
			}
			pnb.add(prices[i]);
		}
		for (i = 1; i < pnb.size(); i += 2) {
			int max1 = getMaxProfit(pnb, 0, i);
			int max2 = getMaxProfit(pnb, i + 1, pnb.size() - 1);
			profit = Math.max(profit, max1 + max2);
		}
		return profit;
	}

	private int getMaxProfit(ArrayList<Integer> pnb, int start, int end) {
		int max = 0;
		if (start < pnb.size() && start < end) {
			int min = pnb.get(start);
			for (int i = start; i <= end; i++) {
				min = Math.min(min, pnb.get(i));
				max = Math.max(max, pnb.get(i) - min);
			}
		}
		return max;
	}

	/*
	 * A forward scan to record max profit [0, i], then a reverse scan to record
	 * max profit [i, n - 1].
	 */
	public int maxProfit3(int[] prices) {
		if (prices == null || prices.length < 2) {
			return 0;
		}
		int len = prices.length;

		int[] t1 = new int[len];
		int min = prices[0], maxp = 0;
		for (int i = 0; i < len; i++) {
			min = Math.min(min, prices[i]);
			maxp = Math.max(maxp, prices[i] - min);
			t1[i] = maxp;
		}

		int max = prices[len - 1];
		int result = 0;
		maxp = 0;
		for (int i = len - 1; i >= 0; i--) {
			max = Math.max(max, prices[i]);
			maxp = Math.max(maxp, max - prices[i]);
			result = Math.max(result, maxp + t1[i]);
		}

		return result;
	}

	/**
	 * $(Pow(x, n))
	 * 
	 * Implement pow(x, n).
	 */

	/*
	 * Watch out for negative n.
	 */
	public double pow(double x, int n) {
		if (x == 0 || x == 1) {
			return x;
		}

		boolean negative = n < 0;
		if (negative) {
			n = ~n + 1;
		}
		double result = powRecursive(x, n);
		if (negative) {
			return 1 / result;
		} else {
			return result;
		}
	}

	private double powRecursive(double x, int n) {
		if (n == 0) {
			return 1;
		}

		double half = pow(x, n / 2);
		if ((n & 1) == 0) {
			return half * half;
		} else {
			return half * half * x;
		}
	}

	/**
	 * $(Edit Distance)
	 * 
	 * Given two words word1 and word2, find the minimum number of steps
	 * required to convert word1 to word2. (each operation is counted as 1
	 * step.)
	 * 
	 * You have the following 3 operations permitted on a word:
	 * 
	 * a) Insert a character b) Delete a character c) Replace a character
	 */

	public int minDistance(String word1, String word2) {
		if (word1 == null || word2 == null) {
			return 0;
		}

		int len1 = word1.length(), len2 = word2.length();
		int[][] dp = new int[len1 + 1][len2 + 1];
		dp[0][0] = 0;
		for (int col = 1; col <= len2; col++) {
			dp[0][col] = col;
		}
		for (int row = 1; row <= len1; row++) {
			dp[row][0] = row;
		}

		for (int row = 1; row <= len1; row++) {
			for (int col = 1; col <= len2; col++) {
				dp[row][col] = Math.min(dp[row - 1][col], dp[row][col - 1]) + 1;
				int adj = (word1.charAt(row - 1) == word2.charAt(col - 1) ? 0
						: 1);
				dp[row][col] = Math.min(dp[row - 1][col - 1] + adj,
						dp[row][col]);
			}
		}

		return dp[len1][len2];
	}

	/**
	 * $(Validate Binary Search Tree)
	 * 
	 * Given a binary tree, determine if it is a valid binary search tree (BST).
	 * 
	 * Assume a BST is defined as follows:
	 * 
	 * The left subtree of a node contains only nodes with keys less than the
	 * node's key. The right subtree of a node contains only nodes with keys
	 * greater than the node's key. Both the left and right subtrees must also
	 * be binary search trees.
	 */

	public boolean isValidBST(TreeNode root) {
		Stack<TreeNode> s = new Stack<TreeNode>();
		TreeNode curr = root;
		boolean stop = false;
		ArrayList<TreeNode> arr = new ArrayList<TreeNode>();

		while (!stop) {
			if (curr != null) {
				s.push(curr);
				curr = curr.left;
			} else {
				if (!s.isEmpty()) {
					curr = s.pop();
					arr.add(curr);
				} else {
					stop = false;
				}
			}
		}

		boolean result = true;
		for (int i = 1; i < arr.size(); i++) {
			if (arr.get(i).val < arr.get(i - 1).val) {
				result = false;
			}
		}
		return result;
	}

	/**
	 * $(Binary Tree Zigzag Level Order Traversal)
	 * 
	 * Given a binary tree, return the zigzag level order traversal of its
	 * nodes' values. (ie, from left to right, then right to left for the next
	 * level and alternate between).
	 * 
	 * For example: Given binary tree {3,9,20,#,#,15,7}, return its zigzag level
	 * order traversal as: [ [3], [20,9], [15,7] ]
	 */

	public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (root == null) {
			return result;
		}

		ArrayList<TreeNode> src = new ArrayList<TreeNode>();
		src.add(root);
		boolean reverse = false;
		while (!src.isEmpty()) {
			ArrayList<TreeNode> dst = new ArrayList<TreeNode>();
			ArrayList<Integer> curr = new ArrayList<Integer>();
			if (reverse) {
				for (int i = src.size() - 1; i >= 0; i--) {
					curr.add(src.get(i).val);
				}
			} else {
				for (int i = 0; i < src.size(); i++) {
					curr.add(src.get(i).val);
				}
			}
			result.add(curr);
			reverse = !reverse;

			for (TreeNode node : src) {
				if (node.left != null) {
					dst.add(node.left);
				}
				if (node.right != null) {
					dst.add(node.right);
				}
			}
			src = dst;
		}

		return result;
	}

	/**
	 * $(Partition List)
	 * 
	 * Given a linked list and a value x, partition it such that all nodes less
	 * than x come before nodes greater than or equal to x.
	 * 
	 * You should preserve the original relative order of the nodes in each of
	 * the two partitions.
	 * 
	 * For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
	 */

	public ListNode partition(ListNode head, int x) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode back = dummy;

		while (back.next != null && back.next.val < x) {
			back = back.next;
		}

		ListNode fptr = back.next, bptr = back;
		while (fptr != null) {
			if (fptr.val < x) {
				bptr.next = fptr.next;
				fptr.next = back.next;
				back.next = fptr;
				back = back.next;
				fptr = bptr.next;
			} else {
				fptr = fptr.next;
				bptr = bptr.next;
			}
		}

		return dummy.next;
	}

	/**
	 * $(Unique Binary Search Trees II)
	 * 
	 * Given n, generate all structurally unique BST's (binary search trees)
	 * that store values 1...n.
	 */

	public ArrayList<TreeNode> generateTrees(int n) {
		return generateTrees(1, n);
	}

	private ArrayList<TreeNode> generateTrees(int start, int end) {
		ArrayList<TreeNode> result = new ArrayList<TreeNode>();
		if (start > end) {
			result.add(null);
			return result;
		}

		for (int i = start; i <= end; i++) {
			ArrayList<TreeNode> leftTrees = generateTrees(start, i - 1);
			ArrayList<TreeNode> rightTrees = generateTrees(i + 1, end);
			for (TreeNode left : leftTrees) {
				for (TreeNode right : rightTrees) {
					TreeNode newRoot = new TreeNode(i);
					newRoot.left = left;
					newRoot.right = right;
					result.add(newRoot);
				}
			}
		}
		return result;
	}

	/**
	 * $(Simplify Path)
	 * 
	 * Given an absolute path for a file (Unix-style), simplify it.
	 * 
	 * For example, path = "/home/", => "/home" path = "/a/./b/../../c/", =>
	 * "/c" click to show corner cases.
	 * 
	 * Corner Cases: Did you consider the case where path = "/../"? In this
	 * case, you should return "/". Another corner case is the path might
	 * contain multiple slashes '/' together, such as "/home//foo/". In this
	 * case, you should ignore redundant slashes and return "/home/foo".
	 */

	public String simplifyPath(String path) {
		if (path == null) {
			return null;
		}

		String[] dirs = path.split("/+");
		if (dirs.length == 0) {
			return "/";
		}

		String[] stack = new String[dirs.length];
		int top = 0;
		for (String dir : dirs) {
			if (".".equals(dir) || dir.isEmpty()) {
				continue;
			} else if ("..".equals(dir)) {
				if (top > 0) {
					top--;
				}
			} else {
				stack[top++] = dir;
			}
		}

		if (top == 0) {
			return "/";
		} else {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < top; i++) {
				sb.append("/");
				sb.append(stack[i]);
			}
			return sb.toString();
		}
	}

	/**
	 * $(Insert Interval)
	 * 
	 * Given a set of non-overlapping intervals, insert a new interval into the
	 * intervals (merge if necessary).
	 * 
	 * You may assume that the intervals were initially sorted according to
	 * their start times.
	 * 
	 * Example 1: Given intervals [1,3],[6,9], insert and merge [2,5] in as
	 * [1,5],[6,9].
	 * 
	 * Example 2: Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9]
	 * in as [1,2],[3,10],[12,16].
	 * 
	 * This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].
	 */

	public ArrayList<Interval> insert(ArrayList<Interval> intervals,
			Interval newInterval) {
		if (intervals == null || newInterval == null) {
			return intervals;
		} else if (intervals.isEmpty()) {
			intervals.add(newInterval);
			return intervals;
		}

		int size = intervals.size();
		int startPos = findPosition(intervals, newInterval.start);
		int endPos = findPosition(intervals, newInterval.end);
		int delStart = startPos, delEnd = endPos - 1;
		if (startPos > 0
				&& newInterval.start <= intervals.get(startPos - 1).end) {
			newInterval.start = intervals.get(startPos - 1).start;
			delStart = startPos - 1;
		}
		if (endPos > 0 && newInterval.end <= intervals.get(endPos - 1).end) {
			newInterval.end = intervals.get(endPos - 1).end;
		} else if (endPos < size
				&& newInterval.end == intervals.get(endPos).start) {
			newInterval.end = intervals.get(endPos).end;
			delEnd = endPos;
		}
		for (int i = 0; i < delEnd - delStart + 1; i++) {
			intervals.remove(delStart);
		}
		intervals.add(delStart, newInterval);

		return intervals;
	}

	private int findPosition(ArrayList<Interval> arr, int val) {
		int start = 0, end = arr.size() - 1;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			if (arr.get(mid).start == val) {
				return mid;
			} else if (arr.get(mid).start < val) {
				start = mid + 1;
			} else {
				end = mid - 1;
			}
		}
		return start;
	}

	/**
	 * $(Letter Combinations of a Phone Number)
	 * 
	 * Given a digit string, return all possible letter combinations that the
	 * number could represent.
	 * 
	 * A mapping of digit to letters (just like on the telephone buttons) is
	 * given below. 1:[], 2:[abc], 3:[def], 4:[ghi], 5:[jkl], 6:[mno], 7:[pqrs],
	 * 8:[tuv], 9:[wxyz]
	 * 
	 * Input:Digit string "23" Output: ["ad", "ae", "af", "bd", "be", "bf",
	 * "cd", "ce", "cf"]. Note: Although the above answer is in lexicographical
	 * order, your answer could be in any order you want.
	 */

	public ArrayList<String> letterCombinations(String digits) {
		ArrayList<String> result = new ArrayList<String>();
		if (digits == null || !digits.matches("\\d*")) {
			return result;
		}

		String[] dict = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs",
				"tuv", "wxyz" };
		letterCombinationsDFS(result, "", digits, dict);

		return result;
	}

	private void letterCombinationsDFS(ArrayList<String> result, String sofar,
			String digits, String[] dict) {
		if (digits.isEmpty()) {
			result.add(sofar);
			return;
		}

		char[] candidates = dict[digits.charAt(0) - '0'].toCharArray();
		for (char ch : candidates) {
			letterCombinationsDFS(result, sofar + ch, digits.substring(1), dict);
		}
	}

	/**
	 * $(First Missing Positive)
	 * 
	 * Given an unsorted integer array, find the first missing positive integer.
	 * 
	 * For example, Given [1,2,0] return 3, and [3,4,-1,1] return 2.
	 * 
	 * Your algorithm should run in O(n) time and uses constant space.
	 */

	/*
	 * Use two pointers: left and right, and mantain following invariant:
	 * A[0]...A[left - 1] are ordered from 1 to left, exclude left;
	 */
	public int firstMissingPositive(int[] A) {
		if (A == null || A.length == 0) {
			return 1;
		}
		int len = A.length, left = 0, right = len - 1;
		while (left <= right) {
			if (A[left] == left + 1) {
				left++;
			} else {
				if (A[left] <= len && A[left] > 0 && left <= A[left] - 1
						&& A[A[left] - 1] != A[left]) {
					arraySwap(A, left, A[left] - 1);
				} else {
					arraySwap(A, left, right);
					right--;
				}
			}
		}
		while (left < len && A[left] == left + 1) {
			left++;
		}
		return left + 1;
	}

	/**
	 * $(Sqrt(x))
	 * 
	 * Implement int sqrt(int x).
	 * 
	 * Compute and return the square root of x.
	 */

	public int sqrt(int x) {
		if (x <= 1) {
			return x;
		}
		double guess = (double) x / 2, n = 30;
		while (n-- > 0) {
			guess = (guess + (double) x / guess) / 2;
		}
		int result = Double.valueOf(guess).intValue();
		if (result * result > x) {
			result--;
		}
		return result;
	}

	@Test
	public void testSqrt() {
		System.out.println(sqrt(2147395600));
		System.out.println(sqrt(2));
	}

	/**
	 * $(Combination Sum)
	 * 
	 * Given a set of candidate numbers (C) and a target number (T), find all
	 * unique combinations in C where the candidate numbers sums to T.
	 * 
	 * The same repeated number may be chosen from C unlimited number of times.
	 * 
	 * Note: All numbers (including target) will be positive integers. Elements
	 * in a combination (a1, a2, … , ak) must be in non-descending order. (ie,
	 * a1 ≤ a2 ≤ … ≤ ak). The solution set must not contain duplicate
	 * combinations. For example, given candidate set 2,3,6,7 and target 7, A
	 * solution set is: [7] [2, 2, 3]
	 */

	public ArrayList<ArrayList<Integer>> combinationSum(int[] candidates,
			int target) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (candidates == null) {
			return result;
		}

		Arrays.sort(candidates);
		int[] occurrence = new int[candidates.length];
		combinationSumDFS(result, candidates, occurrence, target, 0);

		return result;
	}

	private void combinationSumDFS(ArrayList<ArrayList<Integer>> result,
			int[] candidates, int[] occurence, int target, int index) {
		if (index == candidates.length) {
			if (target == 0) {
				ArrayList<Integer> combination = new ArrayList<Integer>();
				for (int i = 0; i < occurence.length; i++) {
					int cnt = occurence[i];
					int num = candidates[i];
					for (int j = 0; j < cnt; j++) {
						combination.add(num);
					}
				}
				result.add(combination);
			}
			return;
		}

		int maxrep = target / candidates[index];
		for (int i = 0; i <= maxrep; i++) {
			occurence[index] = i;
			combinationSumDFS(result, candidates, occurence, target - i
					* candidates[index], index + 1);
		}
	}

	/**
	 * $(Combination Sum II)
	 * 
	 * Given a collection of candidate numbers (C) and a target number (T), find
	 * all unique combinations in C where the candidate numbers sums to T.
	 * 
	 * Each number in C may only be used once in the combination.
	 * 
	 * Note: All numbers (including target) will be positive integers. Elements
	 * in a combination (a1, a2, … , ak) must be in non-descending order. (ie,
	 * a1 ≤ a2 ≤ … ≤ ak). The solution set must not contain duplicate
	 * combinations. For example, given candidate set 10,1,2,7,6,1,5 and target
	 * 8, A solution set is: [1, 7] [1, 2, 5] [2, 6] [1, 1, 6]
	 */

	public ArrayList<ArrayList<Integer>> combinationSum2(int[] num, int target) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null) {
			return result;
		}

		Arrays.sort(num);
		ArrayList<Integer> combination = new ArrayList<Integer>();
		boolean[] visited = new boolean[num.length];
		combinationSum2DFS(result, combination, num, target, 0, visited);

		return result;
	}

	private void combinationSum2DFS(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> combination, int[] num, int target, int index,
			boolean[] visited) {
		if (index == num.length) {
			if (target == 0) {
				result.add(new ArrayList<Integer>(combination));
			}
			return;
		}

		if (!visited[index]
				&& (index == 0 || num[index] != num[index - 1] || visited[index - 1])
				&& target - num[index] >= 0) {
			visited[index] = true;
			combination.add(num[index]);
			combinationSum2DFS(result, combination, num, target - num[index],
					index + 1, visited);
			combination.remove(combination.size() - 1);
			visited[index] = false;
		}
		combinationSum2DFS(result, combination, num, target, index + 1, visited);
	}

	/**
	 * $(Jump Game II)
	 * 
	 * Given an array of non-negative integers, you are initially positioned at
	 * the first index of the array.
	 * 
	 * Each element in the array represents your maximum jump length at that
	 * position.
	 * 
	 * Your goal is to reach the last index in the minimum number of jumps.
	 * 
	 * For example: Given array A = [2,3,1,1,4]
	 * 
	 * The minimum number of jumps to reach the last index is 2. (Jump 1 step
	 * from index 0 to 1, then 3 steps to the last index.)
	 */

	/*
	 * greedy solution
	 */
	public int jump(int[] A) {
		if (A == null || A.length <= 1) {
			return 0;
		}

		int len = A.length, jump = 1;
		int start = 0, end = A[start];
		int nextStart = -1, nextEnd = -1;
		while (true) {
			if (end >= len - 1) {
				break;
			}
			for (int i = end; i > start; i--) {
				if (i + A[i] > nextEnd) {
					nextStart = i;
					nextEnd = i + A[i];
				}
			}
			start = nextStart;
			end = nextEnd;
			jump++;
		}

		return jump;
	}

	/*
	 * sequence DP, dp[i] denotes the least step to reach i. To get dp[i], find
	 * a position j, 0 <= j < i, if we can jump from j to i in one step, and j
	 * is reachable from 0, then the least step needed to reach i is dp[j] + 1.
	 * The trikey part is that we iterate from 0 to i to find j and we have to
	 * stop if we find such j to save time, because the step from 0 to j will
	 * only increase as j increases.
	 */
	public int jumpDP(int[] A) {
		if (A == null || A.length == 0) {
			return 0;
		}

		int len = A.length;
		int[] dp = new int[len];
		dp[0] = 0;
		for (int i = 1; i < len; i++) {
			dp[i] = Integer.MAX_VALUE;
			for (int j = 0; j < i; j++) {
				if (j + A[j] >= i && dp[j] != Integer.MAX_VALUE) {
					dp[i] = dp[j] + 1;
					break;
				}
			}
		}

		return dp[len - 1];
	}

	/**
	 * $(Scramble String)
	 * 
	 * Given a string s1, we may represent it as a binary tree by partitioning
	 * it to two non-empty substrings recursively. To scramble the string, we
	 * may choose any non-leaf node and swap its two children. Given two strings
	 * s1 and s2 of the same length, determine if s2 is a scrambled string of
	 * s1.
	 */

	public boolean isScramble(String s1, String s2) {
		if (s1 == null || s2 == null || s1.length() != s2.length()) {
			return false;
		}

		int len = s1.length();
		char[] map1 = new char[256], map2 = new char[256];
		for (int i = 0; i < len; i++) {
			map1[s1.charAt(i)]++;
			map2[s2.charAt(i)]++;
		}
		for (int i = 0; i < 256; i++) {
			if (map1[i] != map2[i]) {
				return false;
			}
		}
		if (len <= 3) {
			return true;
		}

		for (int i = 1; i < len; i++) {
			if (isScramble(s1.substring(0, i), s2.substring(0, i))
					&& isScramble(s1.substring(i, len), s2.substring(i, len))
					|| isScramble(s1.substring(0, i),
							s2.substring(len - i, len))
					&& isScramble(s1.substring(i, len),
							s2.substring(0, len - i))) {
				return true;
			}
		}

		return false;
	}

	/**
	 * $(Longest Valid Parentheses)
	 * 
	 * Given a string containing just the characters '(' and ')', find the
	 * length of the longest valid (well-formed) parentheses substring.
	 * 
	 * For "(()", the longest valid parentheses substring is "()", which has
	 * length = 2.
	 * 
	 * Another example is ")()())", where the longest valid parentheses
	 * substring is "()()", which has length = 4.
	 */

	public int longestValidParentheses(String s) {
		if (s == null) {
			return 0;
		}

		int cnt = 0, open = 0, max = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				open++;
			} else {
				if (open == 0) {
					cnt = 0;
				} else {
					open--;
					cnt += 2;
					if (open == 0) {
						max = Math.max(max, cnt);
					}
				}
			}
		}

		cnt = 0;
		open = 0;
		for (int i = s.length() - 1; i >= 0; i--) {
			if (s.charAt(i) == ')') {
				open++;
			} else {
				if (open == 0) {
					cnt = 0;
				} else {
					open--;
					cnt += 2;
					if (open == 0) {
						max = Math.max(max, cnt);
					}
				}
			}
		}

		return max;
	}

	/**
	 * $(Convert Sorted List to Binary Search Tree)
	 * 
	 * Given a singly linked list where elements are sorted in ascending order,
	 * convert it to a height balanced BST.
	 */

	public TreeNode sortedListToBST(ListNode head) {
		int len = 0;
		ListNode ptr = head;
		while (ptr != null) {
			len++;
			ptr = ptr.next;
		}

		return sortedListToBST(head, 0, len - 1);
	}

	private TreeNode sortedListToBST(ListNode head, int start, int end) {
		if (start > end) {
			return null;
		}

		int mid = start + (end - start) / 2, count = mid - start;
		ListNode ptr = head;
		for (int i = 0; i < count; i++) {
			ptr = ptr.next;
		}
		TreeNode root = new TreeNode(ptr.val);
		root.left = sortedListToBST(head, start, mid - 1);
		root.right = sortedListToBST(ptr.next, mid + 1, end);
		return root;
	}

	/**
	 * $(Reverse Linked List II)
	 * 
	 * Reverse a linked list from position m to n. Do it in-place and in
	 * one-pass.
	 * 
	 * For example: Given 1->2->3->4->5->NULL, m = 2 and n = 4,
	 * 
	 * return 1->4->3->2->5->NULL.
	 * 
	 * Note: Given m, n satisfy the following condition: 1 ≤ m ≤ n ≤ length of
	 * list.
	 */

	public ListNode reverseBetween(ListNode head, int m, int n) {
		int num = n - m;
		if (num == 0) {
			return head;
		}

		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode front = dummy;
		while (n > 0) {
			front = front.next;
			n--;
		}

		ListNode fp = dummy.next, bp = dummy;
		while (m > 1) {
			fp = fp.next;
			bp = bp.next;
			m--;
		}

		while (num > 0) {
			bp.next = fp.next;
			fp.next = front.next;
			front.next = fp;
			fp = bp.next;
			num--;
		}

		return dummy.next;
	}

	/**
	 * $(Construct Binary Tree from Preorder and Inorder Traversal)
	 * 
	 * Given preorder and inorder traversal of a tree, construct the binary
	 * tree.
	 * 
	 * Note: You may assume that duplicates do not exist in the tree.
	 */

	public TreeNode buildTree(int[] preorder, int[] inorder) {
		if (preorder == null || inorder == null
				|| preorder.length != inorder.length) {
			return null;
		}
		Map<Integer, Integer> positions = new HashMap<Integer, Integer>();
		for (int i = 0; i < inorder.length; i++) {
			positions.put(inorder[i], i);
		}

		return buildTree(preorder, inorder, 0, preorder.length - 1, 0,
				inorder.length - 1, positions);
	}

	private TreeNode buildTree(int[] preorder, int[] inorder, int preStart,
			int preEnd, int inStart, int inEnd, Map<Integer, Integer> positions) {
		if (inStart > inEnd) {
			return null;
		}

		TreeNode root = new TreeNode(preorder[preStart]);
		int i = positions.get(root.val);
		root.left = buildTree(preorder, inorder, preStart + 1, preStart + i
				- inStart, inStart, i - 1, positions);
		root.right = buildTree(preorder, inorder, preStart + i - inStart + 1,
				preEnd, i + 1, inEnd, positions);
		return root;
	}

	/**
	 * $(Construct Binary Tree from Inorder and Postorder Traversal)
	 * 
	 * Given inorder and postorder traversal of a tree, construct the binary
	 * tree.
	 * 
	 * Note: You may assume that duplicates do not exist in the tree.
	 */

	public TreeNode buildTree2(int[] inorder, int[] postorder) {
		if (inorder == null || postorder == null
				|| inorder.length != postorder.length) {
			return null;
		}

		Map<Integer, Integer> positions = new HashMap<Integer, Integer>();
		for (int i = 0; i < inorder.length; i++) {
			positions.put(inorder[i], i);
		}
		return buildTree2(inorder, 0, inorder.length - 1, postorder, 0,
				postorder.length - 1, positions);
	}

	private TreeNode buildTree2(int[] inorder, int inStart, int inEnd,
			int[] postorder, int pStart, int pEnd,
			Map<Integer, Integer> positions) {
		if (inStart > inEnd) {
			return null;
		}
		TreeNode root = new TreeNode(postorder[pEnd]);
		int i = positions.get(root.val);
		root.left = buildTree2(inorder, inStart, i - 1, postorder, pStart,
				pStart + i - inStart - 1, positions);
		root.right = buildTree2(inorder, i + 1, inEnd, postorder, pStart + i
				- inStart, pEnd - 1, positions);
		return root;
	}

	/**
	 * $(Palindrome Partitioning)
	 * 
	 * Given a string s, partition s such that every substring of the partition
	 * is a palindrome.
	 * 
	 * Return all possible palindrome partitioning of s.
	 * 
	 * For example, given s = "aab", Return
	 * 
	 * [ ["aa","b"], ["a","a","b"] ]
	 */

	public ArrayList<ArrayList<String>> partition(String s) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (s == null) {
			return result;
		} else if (s.isEmpty()) {
			ArrayList<String> partition = new ArrayList<String>();
			partition.add(s);
			result.add(partition);
			return result;
		}

		boolean[][] dp = getPalindromes(s);
		ArrayList<String> partition = new ArrayList<String>();
		partitionDFS(result, partition, dp, s, 0);

		return result;
	}

	private boolean[][] getPalindromes(String s) {
		int len = s.length();
		boolean[][] dp = new boolean[len][len];

		for (int row = len - 1; row >= 0; row--) {
			for (int col = row; col < len; col++) {
				if (row == col) {
					dp[row][col] = true;
				} else if (col - row == 1) {
					dp[row][col] = s.charAt(row) == s.charAt(col);
				} else {
					dp[row][col] = s.charAt(row) == s.charAt(col)
							&& dp[row + 1][col - 1];
				}
			}
		}

		return dp;
	}

	private void partitionDFS(ArrayList<ArrayList<String>> result,
			ArrayList<String> partition, boolean[][] dp, String s, int row) {
		if (row == dp.length) {
			result.add(new ArrayList<String>(partition));
			return;
		}

		for (int col = row; col < dp.length; col++) {
			if (dp[row][col]) {
				partition.add(s.substring(row, col + 1));
				partitionDFS(result, partition, dp, s, col + 1);
				partition.remove(partition.size() - 1);
			}
		}
	}

	/**
	 * $(Palindrome Partitioning II)
	 * 
	 * Given a string s, partition s such that every substring of the partition
	 * is a palindrome.
	 * 
	 * Return the minimum cuts needed for a palindrome partitioning of s.
	 * 
	 * For example, given s = "aab", Return 1 since the palindrome partitioning
	 * ["aa","b"] could be produced using 1 cut.
	 */

	/*
	 * simple implementation
	 */
	public int minCut(String s) {
		if (s == null || s.isEmpty()) {
			return 0;
		}

		int len = s.length();
		boolean[][] palindromeMap = getPalindromes(s);
		int[] dp = new int[len + 1];
		dp[0] = 0;
		for (int i = 1; i <= len; i++) {
			int minCut = Integer.MAX_VALUE;
			for (int j = 0; j < i; j++) {
				if (palindromeMap[j][i - 1]) {
					if (j == 0) {
						minCut = 0;
						break;
					} else {
						minCut = Math.min(minCut, dp[j] + 1);
					}
				}
			}
			dp[i] = minCut;
		}

		return dp[len];
	}

	/*
	 * merge two dp
	 */
	public int minCut2(String s) {
		if (s == null || s.length() < 2) {
			return 0;
		}

		int len = s.length();

		boolean[][] m = new boolean[len][len];
		int[] map = new int[len];
		for (int i = 0; i < len; i++) {
			map[i] = len - 1;
		}
		map[len - 1] = 0;
		for (int row = len - 1; row >= 0; row--) {
			for (int col = row; col < len; col++) {
				if (s.charAt(row) == s.charAt(col)
						&& (col - row < 2 || m[row + 1][col - 1])) {
					m[row][col] = true;
					if (col == len - 1) {
						map[row] = 0;
					} else {
						map[row] = Math.min(map[row], 1 + map[col + 1]);
					}
				}
			}
		}

		return map[0];
	}

	/**
	 * $(Reverse Nodes in k-Group)
	 * 
	 * Given a linked list, reverse the nodes of a linked list k at a time and
	 * return its modified list.
	 * 
	 * If the number of nodes is not a multiple of k then left-out nodes in the
	 * end should remain as it is.
	 * 
	 * You may not alter the values in the nodes, only nodes itself may be
	 * changed.
	 * 
	 * Only constant memory is allowed.
	 * 
	 * For example, Given this linked list: 1->2->3->4->5
	 * 
	 * For k = 2, you should return: 2->1->4->3->5
	 * 
	 * For k = 3, you should return: 3->2->1->4->5
	 */

	public ListNode reverseKGroup(ListNode head, int k) {
		ListNode dummy = new ListNode(0), ptr = head, bptr = dummy, fptr = head;
		dummy.next = head;
		while (true) {
			for (int i = 1; i < k && ptr != null; i++) {
				ptr = ptr.next;
			}
			if (ptr == null) {
				break;
			}
			for (int i = 1; i < k; i++) {
				bptr.next = fptr.next;
				fptr.next = ptr.next;
				ptr.next = fptr;
				fptr = bptr.next;
			}
			for (int i = 0; i < k; i++) {
				ptr = ptr.next;
				fptr = fptr.next;
				bptr = bptr.next;
			}
		}

		return dummy.next;
	}

	/**
	 * $(Longest Substring Without Repeating Characters)
	 * 
	 * Given a string, find the length of the longest substring without
	 * repeating characters. For example, the longest substring without
	 * repeating letters for "abcabcbb" is "abc", which the length is 3. For
	 * "bbbbb" the longest substring is "b", with the length of 1.
	 */

	public int lengthOfLongestSubstring(String s) {
		if (s == null || s.length() == 0) {
			return 0;
		}
		HashMap<Character, Integer> m = new HashMap<Character, Integer>();
		int len = s.length();
		int i = 0, max = 0;
		while (i < len) {
			Character c = s.charAt(i);
			if (m.containsKey(c)) {
				i = m.get(c) + 1;
				max = Math.max(max, m.size());
				m.clear();
			} else {
				m.put(c, i);
				i++;
			}
		}
		max = Math.max(max, m.size());
		return max;
	}

	/**
	 * $(Anagrams)
	 * 
	 * Given an array of strings, return all groups of strings that are
	 * anagrams.
	 * 
	 * Note: All inputs will be in lower-case.
	 */

	public ArrayList<String> anagrams(String[] strs) {
		ArrayList<String> result = new ArrayList<String>();
		if (strs != null && strs.length > 0) {
			HashMap<String, ArrayList<String>> m = new HashMap<String, ArrayList<String>>();
			for (int i = 0; i < strs.length; i++) {
				int[] map = new int[26];
				String s = strs[i];
				for (int j = 0; j < s.length(); j++) {
					map[s.charAt(j) - 'a']++;
				}
				StringBuilder sig = new StringBuilder();
				for (int j = 0; j < map.length; j++) {
					if (map[j] > 0) {
						sig.append('a' + j);
						sig.append(map[j]);
					}
				}
				String sigStr = sig.toString();
				ArrayList<String> list = m.get(sigStr);
				if (list == null) {
					ArrayList<String> newList = new ArrayList<String>();
					newList.add(s);
					m.put(sigStr, newList);
				} else {
					list.add(s);
				}
			}
			for (String key : m.keySet()) {
				ArrayList<String> list = m.get(key);
				if (list.size() > 1) {
					for (String str : list) {
						result.add(str);
					}
				}
			}
		}
		return result;
	}

	/**
	 * $(ZigZag Conversion)
	 * 
	 * The string "PAYPALISHIRING" is written in a zigzag pattern on a given
	 * number of rows like this: (you may want to display this pattern in a
	 * fixed font for better legibility)
	 * 
	 * And then read line by line: "PAHNAPLSIIGYIR" Write the code that will
	 * take a string and make this conversion given a number of rows: string
	 * convert(string text, int nRows); convert("PAYPALISHIRING", 3) should
	 * return "PAHNAPLSIIGYIR".
	 * 
	 */

	public String convert(String s, int nRows) {
		if (s == null || s.length() == 0 || nRows < 2) {
			return s;
		}
		int len = s.length(), gpsize = nRows * 2 - 2;
		StringBuilder result = new StringBuilder();
		for (int i = 0; i < nRows; i++) {
			int index;
			if (i == 0 || i == nRows - 1) {
				int n = 0;
				while (true) {
					index = i + n * gpsize;
					if (index >= len) {
						break;
					} else {
						result.append(s.charAt(index));
						n++;
					}
				}
			} else {
				int n = 0, gap = 2 * (nRows - 1 - i);
				while (true) {
					index = i + (n / 2) * gpsize;
					if ((n & 1) == 1) {
						index += gap;
					}
					if (index >= len) {
						break;
					} else {
						result.append(s.charAt(index));
						n++;
					}
				}
			}
		}
		return result.toString();
	}

	@Test
	public void testConvert() {
		String s = "PAYPALISHIRING";
		System.out.println(convert(s, 3));
	}

	/**
	 * $(Longest Palindromic Substring)
	 * 
	 * Given a string S, find the longest palindromic substring in S. You may
	 * assume that the maximum length of S is 1000, and there exists one unique
	 * longest palindromic substring.
	 */

	public String longestPalindrome(String s) {
		if (s == null || s.length() < 2) {
			return s;
		}
		StringBuilder sb = new StringBuilder();
		sb.append(s.charAt(0));
		for (int i = 1; i < s.length(); i++) {
			sb.append("*");
			sb.append(s.charAt(i));
		}
		int len = sb.length(), max = 0;
		int[] pair = new int[2];
		for (int i = 0; i < len; i++) {
			int left, right;
			if ((i & 1) == 1) {
				left = i - 1;
				right = i + 1;
			} else {
				left = i - 2;
				right = i + 2;
			}
			while (left >= 0 && right < len) {
				if (sb.charAt(left) != sb.charAt(right)) {
					break;
				} else {
					if (right - left + 1 > max) {
						max = right - left + 1;
						pair[0] = left / 2;
						pair[1] = right / 2 + 1;
					}
					left -= 2;
					right += 2;
				}
			}
		}
		return s.substring(pair[0], pair[1]);
	}

	/*
	 * DP solution
	 */
	public String longestPalindromeDP(String s) {
		if (s == null || s.isEmpty()) {
			return s;
		}

		int len = s.length();
		int maxLen = 0;
		int[] position = new int[2];
		boolean[][] dp = new boolean[len][len];
		for (int row = len - 1; row >= 0; row--) {
			for (int col = row; col < len; col++) {
				if (row == col) {
					dp[row][col] = true;
				} else if (col - row == 1 && s.charAt(row) == s.charAt(col)) {
					dp[row][col] = true;
				} else {
					dp[row][col] = dp[row + 1][col - 1]
							&& s.charAt(row) == s.charAt(col);
				}

				if (dp[row][col]) {
					int currLen = col - row + 1;
					if (currLen > maxLen) {
						maxLen = currLen;
						position[0] = row;
						position[1] = col + 1;
					}
				}
			}
		}

		return s.substring(position[0], position[1]);
	}

	/**
	 * $(Recover Binary Search Tree)
	 * 
	 * Two elements of a binary search tree (BST) are swapped by mistake.
	 * 
	 * Recover the tree without changing its structure.
	 * 
	 * Note: A solution using O(n) space is pretty straight forward. Could you
	 * devise a constant space solution?
	 */

	/*
	 * O(n) space
	 */
	public void recoverTree(TreeNode root) {
		if (root == null || root.left == null && root.right == null) {
			return;
		}

		ArrayList<TreeNode> arr = new ArrayList<TreeNode>();
		Stack<TreeNode> s = new Stack<TreeNode>();
		TreeNode curr = root;
		boolean stop = false;
		while (!stop) {
			if (curr != null) {
				s.push(curr);
				curr = curr.left;
			} else {
				if (!s.isEmpty()) {
					curr = s.pop();
					arr.add(curr);
					curr = curr.right;
				} else {
					stop = true;
				}
			}
		}

		TreeNode left = null, right = null;
		int size = arr.size();
		for (int i = 0; i < size - 1; i++) {
			if (arr.get(i).val > arr.get(i + 1).val) {
				left = arr.get(i);
				break;
			}
		}
		for (int i = size - 1; i > 0; i--) {
			if (arr.get(i).val < arr.get(i - 1).val) {
				right = arr.get(i);
				break;
			}
		}

		left.val ^= right.val;
		right.val ^= left.val;
		left.val ^= right.val;
	}

	/*
	 * O(1) space
	 */
	public void recoverTree2(TreeNode root) {
		if (root == null || root.left == null && root.right == null) {
			return;
		}

		TreeNode left = null, right = null, last = null, curr = root;
		Stack<TreeNode> stack = new Stack<TreeNode>();
		boolean stop = false;
		while (!stop) {
			if (curr != null) {
				stack.push(curr);
				curr = curr.left;
			} else {
				if (!stack.isEmpty()) {
					curr = stack.pop();
					if (last != null) {
						if (last.val > curr.val) {
							if (left == null) {
								left = last;
								right = curr;
							} else {
								right = curr;
							}
						}
					}
					last = curr;
					curr = curr.right;
				} else {
					stop = true;
				}
			}
		}

		if (left != null && right != null) {
			left.val ^= right.val;
			right.val ^= left.val;
			left.val ^= right.val;
		}
	}

	/**
	 * $(Clone Graph)
	 * 
	 * Clone an undirected graph. Each node in the graph contains a label and a
	 * list of its neighbors.
	 * 
	 * 
	 * OJ's undirected graph serialization: Nodes are labeled uniquely.
	 * 
	 * We use # as a separator for each node, and , as a separator for node
	 * label and each neighbor of the node. As an example, consider the
	 * serialized graph {0,1,2#1,2#2,2}.
	 * 
	 * The graph has a total of three nodes, and therefore contains three parts
	 * as separated by #.
	 * 
	 * First node is labeled as 0. Connect node 0 to both nodes 1 and 2. Second
	 * node is labeled as 1. Connect node 1 to node 2. Third node is labeled as
	 * 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
	 */

	class UndirectedGraphNode {
		int label;
		ArrayList<UndirectedGraphNode> neighbors;

		UndirectedGraphNode(int x) {
			label = x;
			neighbors = new ArrayList<UndirectedGraphNode>();
		}
	};

	/*
	 * first DFS, then clone neighbors
	 */
	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		if (node == null) {
			return null;
		}

		HashMap<Integer, UndirectedGraphNode> map = new HashMap<Integer, UndirectedGraphNode>();
		map.put(node.label, node);
		DFSgraph(node, map);

		HashMap<Integer, UndirectedGraphNode> newMap = new HashMap<Integer, UndirectedGraphNode>();
		for (Integer label : map.keySet()) {
			UndirectedGraphNode oriNode = map.get(label);
			UndirectedGraphNode newNode = newMap.get(label);
			if (newNode == null) {
				newNode = new UndirectedGraphNode(label);
				newMap.put(label, newNode);
			}
			ArrayList<UndirectedGraphNode> newNgb = newNode.neighbors;
			for (UndirectedGraphNode neighbor : oriNode.neighbors) {
				UndirectedGraphNode ngb = newMap.get(neighbor.label);
				if (ngb == null) {
					ngb = new UndirectedGraphNode(neighbor.label);
					newMap.put(ngb.label, ngb);
				}
				newNgb.add(ngb);
			}
		}

		return newMap.get(node.label);
	}

	private void DFSgraph(UndirectedGraphNode node,
			HashMap<Integer, UndirectedGraphNode> map) {
		for (UndirectedGraphNode ngb : node.neighbors) {
			if (!map.containsKey(ngb.label)) {
				map.put(ngb.label, ngb);
				DFSgraph(ngb, map);
			}
		}
	}

	/*
	 * first BFS, then colone neighbors, one line difference from previous
	 * method.
	 */
	public UndirectedGraphNode cloneGraph2(UndirectedGraphNode node) {
		if (node == null) {
			return null;
		}

		HashMap<Integer, UndirectedGraphNode> map = new HashMap<Integer, UndirectedGraphNode>();
		BFSgraph(node, map);

		HashMap<Integer, UndirectedGraphNode> newMap = new HashMap<Integer, UndirectedGraphNode>();
		for (Integer label : map.keySet()) {
			UndirectedGraphNode oriNode = map.get(label);
			UndirectedGraphNode newNode = newMap.get(label);
			if (newNode == null) {
				newNode = new UndirectedGraphNode(label);
				newMap.put(label, newNode);
			}
			ArrayList<UndirectedGraphNode> newNgb = newNode.neighbors;
			for (UndirectedGraphNode neighbor : oriNode.neighbors) {
				UndirectedGraphNode ngb = newMap.get(neighbor.label);
				if (ngb == null) {
					ngb = new UndirectedGraphNode(neighbor.label);
					newMap.put(ngb.label, ngb);
				}
				newNgb.add(ngb);
			}
		}

		return newMap.get(node.label);
	}

	private void BFSgraph(UndirectedGraphNode node,
			HashMap<Integer, UndirectedGraphNode> map) {
		Queue<UndirectedGraphNode> q = new LinkedList<UndirectedGraphNode>();
		q.add(node);
		while (!q.isEmpty()) {
			UndirectedGraphNode curr = q.remove();
			if (map.containsKey(curr.label)) {
				continue;
			}
			map.put(curr.label, curr);

			for (UndirectedGraphNode ngb : curr.neighbors) {
				if (!map.containsKey(ngb.label)) {
					q.add(ngb);
				}
			}
		}
	}

	/*
	 * clone as you BFS
	 */
	public UndirectedGraphNode cloneGraphBFS(UndirectedGraphNode node) {
		if (node == null) {
			return null;
		}
		Queue<UndirectedGraphNode> q1 = new LinkedList<UndirectedGraphNode>();
		Queue<UndirectedGraphNode> q2 = new LinkedList<UndirectedGraphNode>();
		HashMap<Integer, UndirectedGraphNode> m = new HashMap<Integer, UndirectedGraphNode>();
		HashSet<UndirectedGraphNode> v = new HashSet<UndirectedGraphNode>();
		q1.add(node);
		UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
		q2.add(newNode);
		m.put(newNode.label, newNode);
		while (!q1.isEmpty()) {
			UndirectedGraphNode ptr1 = q1.remove();
			UndirectedGraphNode ptr2 = q2.remove();
			if (!v.contains(ptr1)) {
				v.add(ptr1);
				for (UndirectedGraphNode nbr1 : ptr1.neighbors) {
					UndirectedGraphNode nbr2 = m.get(nbr1.label);
					if (nbr2 == null) {
						nbr2 = new UndirectedGraphNode(nbr1.label);
						m.put(nbr2.label, nbr2);
					}
					ptr2.neighbors.add(nbr2);
					if (!v.contains(nbr1)) {
						q1.add(nbr1);
						q2.add(nbr2);
					}
				}
			}
		}
		return newNode;
	}

	/*
	 * clone as you DFS
	 */
	public UndirectedGraphNode cloneGraphDFS(UndirectedGraphNode node) {
		if (node == null) {
			return null;
		}
		HashMap<Integer, UndirectedGraphNode> m = new HashMap<Integer, UndirectedGraphNode>();
		HashSet<UndirectedGraphNode> v = new HashSet<UndirectedGraphNode>();
		UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
		m.put(newNode.label, newNode);
		cloneGraphDFS(node, newNode, m, v);
		return newNode;
	}

	private void cloneGraphDFS(UndirectedGraphNode node,
			UndirectedGraphNode newNode,
			HashMap<Integer, UndirectedGraphNode> m,
			HashSet<UndirectedGraphNode> v) {
		v.add(node);
		for (UndirectedGraphNode nbr1 : node.neighbors) {
			UndirectedGraphNode nbr2 = m.get(nbr1.label);
			if (nbr2 == null) {
				nbr2 = new UndirectedGraphNode(nbr1.label);
				m.put(nbr2.label, nbr2);
			}
			newNode.neighbors.add(nbr2);
			if (!v.contains(nbr1)) {
				cloneGraphDFS(nbr1, nbr2, m, v);
			}
		}
	}

	/**
	 * $(Merge k Sorted Lists)
	 * 
	 * Merge k sorted linked lists and return it as one sorted list. Analyze and
	 * describe its complexity.
	 */

	public ListNode mergeKLists(ArrayList<ListNode> lists) {
		if (lists == null) {
			return null;
		}

		PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(100,
				new ListNodeCmp());
		for (ListNode head : lists) {
			if (head != null) {
				pq.add(head);
			}
		}

		ListNode dummy = new ListNode(0), ptr = dummy;
		while (!pq.isEmpty()) {
			ListNode head = pq.remove();
			ptr.next = head;
			ptr = ptr.next;
			if (head.next != null) {
				pq.add(head.next);
			}
		}

		return dummy.next;
	}

	class ListNodeCmp implements Comparator<ListNode> {
		public int compare(ListNode node1, ListNode node2) {
			return node1.val - node2.val;
		}
	}

	/**
	 * $(Spiral Matrix)
	 * 
	 * Given a matrix of m x n elements (m rows, n columns), return all elements
	 * of the matrix in spiral order.
	 * 
	 * For example, Given the following matrix:
	 * 
	 * [ [ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ] ] You should return
	 * [1,2,3,6,9,8,7,4,5].
	 */

	public ArrayList<Integer> spiralOrder(int[][] matrix) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (matrix != null && matrix.length > 0 && matrix[0] != null
				&& matrix[0].length > 0) {
			int h = matrix.length, w = matrix[0].length;
			int[][] directions = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
			int row = 0, col = 1, direction = 0;
			int upper = 0, right = w - 1, lower = h - 1, left = 0;
			for (int i = 0, r = 0, c = 0; i < h * w; i++) {
				result.add(matrix[r][c]);
				switch (direction) {
				case 0:
					if (c + directions[direction][col] > right) {
						direction = (direction + 1) % 4;
						upper++;
					}
					break;
				case 1:
					if (r + directions[direction][row] > lower) {
						direction = (direction + 1) % 4;
						right--;
					}
					break;
				case 2:
					if (c + directions[direction][col] < left) {
						direction = (direction + 1) % 4;
						lower--;
					}
					break;
				case 3:
					if (r + directions[direction][row] < upper) {
						direction = (direction + 1) % 4;
						left++;
					}
				}
				r += directions[direction][row];
				c += directions[direction][col];
			}
		}
		return result;
	}

	@Test
	public void testSpiralOrder() {
		int h = 6, w = 7;
		int[][] matrix = new int[h][w];
		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) {
				matrix[r][c] = r * w + c + 1;
				System.out.print(matrix[r][c]);
			}
			System.out.println();
		}
		System.out.println("---");
		ArrayList<Integer> spiralOrder = spiralOrder(matrix);
		for (Integer n : spiralOrder) {
			System.out.print(n + ", ");
		}
	}

	/**
	 * $(Word Search)
	 * 
	 * Given a 2D board and a word, find if the word exists in the grid.
	 * 
	 * The word can be constructed from letters of sequentially adjacent cell,
	 * where "adjacent" cells are those horizontally or vertically neighboring.
	 * The same letter cell may not be used more than once.
	 * 
	 * For example, Given board =
	 * 
	 * [ ["ABCE"], ["SFCS"], ["ADEE"] ] word = "ABCCED", -> returns true, word =
	 * "SEE", -> returns true, word = "ABCB", -> returns false.
	 */

	public boolean exist(char[][] board, String word) {
		if (board == null || board.length == 0 || board[0] == null
				|| board[0].length == 0 || word == null) {
			return false;
		}

		int ht = board.length, wd = board[0].length;
		boolean[][] visited = new boolean[ht][wd];
		for (int row = 0; row < ht; row++) {
			for (int col = 0; col < wd; col++) {
				if (board[row][col] == word.charAt(0)) {
					board[row][col] = ' ';
					if (existDFS(board, visited, word.substring(1), row, col)) {
						return true;
					}
					board[row][col] = word.charAt(0);
				}
			}
		}

		return false;
	}

	private boolean existDFS(char[][] board, boolean[][] visited, String word,
			int row, int col) {
		if (word.isEmpty()) {
			return true;
		}

		int[][] directions = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
		for (int i = 0; i < directions.length; i++) {
			int newRow = row + directions[i][0], newCol = col
					+ directions[i][1];
			if (isInBoard(newRow, newCol, board)
					&& board[newRow][newCol] == word.charAt(0)) {
				board[newRow][newCol] = ' ';
				if (existDFS(board, visited, word.substring(1), newRow, newCol)) {
					return true;
				}
				board[newRow][newCol] = word.charAt(0);
			}
		}

		return false;
	}

	private boolean isInBoard(int row, int col, char[][] board) {
		return row >= 0 && row < board.length && col >= 0
				&& col < board[0].length;
	}

	/**
	 * $(Multiply Strings)
	 * 
	 * Given two numbers represented as strings, return multiplication of the
	 * numbers as a string.
	 * 
	 * Note: The numbers can be arbitrarily large and are non-negative.
	 */

	public String multiply(String num1, String num2) {
		if (num1 == null || num1.isEmpty() || num2 == null || num2.isEmpty()
				|| "0".equals(num1) || "0".equals(num2)) {
			return "0";
		}

		int len1 = num1.length(), len2 = num2.length();
		if (len1 < len2) {
			return multiply(num2, num1);
		}

		Set<Character> digits = new HashSet<Character>();
		for (int i = 0; i < len2 && digits.size() < 10; i++) {
			digits.add(num2.charAt(i));
		}
		Map<Character, String> map = getProducts(num1, digits);

		Queue<String> q = new LinkedList<String>();
		StringBuilder sb = new StringBuilder();
		for (int i = num2.length() - 1; i >= 0; i--) {
			if (num2.charAt(i) != '0') {
				String product = map.get(num2.charAt(i));
				product += sb.toString();
				q.add(product);
			}
			sb.append('0');
		}

		while (q.size() > 1) {
			Queue<String> dst = new LinkedList<String>();
			while (!q.isEmpty()) {
				String x1 = q.poll(), x2 = q.poll();
				dst.add(stringAdd(x1, x2));
			}
			q = dst;
		}

		return q.poll();
	}

	private Map<Character, String> getProducts(String num, Set<Character> digits) {
		char[] chs = num.toCharArray();
		int[] nums = new int[chs.length];
		for (int i = 0; i < nums.length; i++) {
			nums[i] = chs[i] - '0';
		}
		HashMap<Character, String> result = new HashMap<Character, String>();
		for (Character digit : digits) {
			if (digit == '0') {
				result.put(digit, "0");
			} else {
				result.put(digit, stringMultiply(nums, digit - '0'));
			}
		}

		return result;
	}

	private String stringMultiply(int[] num, int digit) {
		int carry = 0;
		int[] products = new int[num.length];
		for (int i = num.length - 1; i >= 0; i--) {
			int product = digit * num[i] + carry;
			products[i] = product % 10;
			carry = product / 10;
		}

		StringBuilder sb = new StringBuilder();
		if (carry != 0) {
			sb.append((char) ('0' + carry));
		}
		for (int n : products) {
			sb.append((char) ('0' + n));
		}

		return sb.toString();
	}

	private String stringAdd(String num1, String num2) {
		if (num1 == null) {
			return num2;
		} else if (num2 == null) {
			return num1;
		}

		int len1 = num1.length(), len2 = num2.length();
		if (len1 < len2) {
			return stringAdd(num2, num1);
		}

		int[] arr1 = new int[len1], arr2 = new int[len2];
		for (int i = 0; i < arr1.length; i++) {
			arr1[i] = num1.charAt(i) - '0';
		}
		for (int i = 0; i < arr2.length; i++) {
			arr2[i] = num2.charAt(i) - '0';
		}

		int[] result = new int[len1];
		int carry = 0, i = 0;
		for (; i < len2; i++) {
			int sum = arr1[len1 - 1 - i] + arr2[len2 - 1 - i] + carry;
			result[len1 - 1 - i] = sum % 10;
			carry = sum / 10;
		}

		while (i < len1) {
			int sum = arr1[len1 - 1 - i] + carry;
			result[len1 - 1 - i] = sum % 10;
			carry = sum / 10;
			i++;
		}

		StringBuilder sb = new StringBuilder();
		if (carry > 0) {
			sb.append((char) ('0' + carry));
		}
		for (int digit : result) {
			sb.append((char) ('0' + digit));
		}

		return sb.toString();
	}

	/**
	 * $(Distinct Subsequences)
	 * 
	 * Given a string S and a string T, count the number of distinct
	 * subsequences of T in S.
	 * 
	 * A subsequence of a string is a new string which is formed from the
	 * original string by deleting some (can be none) of the characters without
	 * disturbing the relative positions of the remaining characters. (ie, "ACE"
	 * is a subsequence of "ABCDE" while "AEC" is not).
	 * 
	 * Here is an example: S = "rabbbit", T = "rabbit"
	 * 
	 * Return 3.
	 */

	/*
	 * Not working, TLE
	 */
	public int numDistinctTopDown(String S, String T) {
		if (S == null || S.isEmpty() || T == null || T.isEmpty()
				|| T.length() > S.length()) {
			return 0;
		}

		int lenS = S.length(), lenT = T.length();
		HashMap<Character, ArrayList<Integer>> m = new HashMap<Character, ArrayList<Integer>>();
		HashSet<Character> v = new HashSet<Character>();
		for (int i = 0; i < lenT; i++) {
			char ch = T.charAt(i);
			if (v.contains(ch)) {
				continue;
			} else {
				v.add(ch);
			}
			boolean found = false;
			for (int j = 0; j < lenS; j++) {
				if (S.charAt(j) == ch) {
					found = true;
					ArrayList<Integer> pos = m.get(ch);
					if (pos == null) {
						pos = new ArrayList<Integer>();
						m.put(ch, pos);
					}
					pos.add(j);
				}
			}
			if (!found) {
				return 0;
			}
		}

		for (Character ch : m.keySet()) {
			System.out.println(ch + " : " + m.get(ch));
		}

		int[] result = new int[1];
		numDistinctDFS(result, m, T, -1);
		return result[0];
	}

	private void numDistinctDFS(int[] result,
			HashMap<Character, ArrayList<Integer>> m, String str, int index) {
		if (str.isEmpty()) {
			result[0]++;
			return;
		}

		ArrayList<Integer> pos = m.get(str.charAt(0));
		int i = nextAvailable(pos, index);
		if (i > -1) {
			for (; i < pos.size(); i++) {
				numDistinctDFS(result, m, str.substring(1), pos.get(i));
			}
		}
	}

	private int nextAvailable(ArrayList<Integer> pos, int index) {
		if (index < pos.get(pos.size() - 1)) {
			int start = 0, end = pos.size() - 1;
			while (start <= end) {
				int mid = (start + end) / 2, midVal = pos.get(mid);
				if (index == midVal) {
					return mid + 1;
				} else if (index < midVal) {
					end = mid - 1;
				} else {
					start = mid + 1;
				}
			}
			return start;
		}
		return -1;
	}

	/*
	 * DP solution
	 */
	public int numDistinct(String S, String T) {
		if (S == null || S.isEmpty() || T == null || T.isEmpty()
				|| T.length() > S.length()) {
			return 0;
		}

		int lenS = S.length(), lenT = T.length();
		int[][] dp = new int[lenT][lenS];

		for (int col = 0; col < lenS; col++) {
			if (S.charAt(col) == T.charAt(0)) {
				dp[0][col] = (col == 0 ? 1 : dp[0][col - 1] + 1);
			} else {
				dp[0][col] = (col == 0 ? 0 : dp[0][col - 1]);
			}
		}
		for (int row = 1; row < lenT; row++) {
			dp[row][0] = 0;
		}

		for (int row = 1; row < lenT; row++) {
			for (int col = row; col < lenS; col++) {
				dp[row][col] = dp[row][col - 1];
				if (S.charAt(col) == T.charAt(row)) {
					dp[row][col] += dp[row - 1][col - 1];
				}
			}
		}

		return dp[lenT - 1][lenS - 1];
	}

	/**
	 * $(Binary Tree Maximum Path Sum)
	 * 
	 * Given a binary tree, find the maximum path sum. The path may start and
	 * end at any node in the tree. For example: Given the below binary tree,
	 * [1, 2, 3]. Return 6.
	 */

	public int maxPathSum(TreeNode root) {
		int[] max = { Integer.MIN_VALUE };
		maxPathSumDFS(root, max);
		return max[0];
	}

	public int maxPathSumDFS(TreeNode root, int[] max) {
		if (root == null) {
			return 0;
		}

		int left = maxPathSumDFS(root.left, max), right = maxPathSumDFS(
				root.right, max);
		int currMax = Math.max(Math.max(left, right), 0) + root.val;
		max[0] = Math.max(max[0], Math.max(currMax, left + right + root.val));
		return currMax;
	}

	/**
	 * $(Word Break)
	 * 
	 * Given a string s and a dictionary of words dict, determine if s can be
	 * segmented into a space-separated sequence of one or more dictionary
	 * words.
	 * 
	 * For example, given s = "leetcode", dict = ["leet", "code"].
	 * 
	 * Return true because "leetcode" can be segmented as "leet code".
	 */

	public boolean wordBreakBFS(String s, Set<String> dict) {
		if (s == null || dict == null) {
			return false;
		}

		HashMap<Character, ArrayList<String>> m = new HashMap<Character, ArrayList<String>>();
		HashSet<Character> setDict = new HashSet<Character>();
		for (String word : dict) {
			ArrayList<String> arr = m.get(word.charAt(0));
			if (arr == null) {
				arr = new ArrayList<String>();
				m.put(word.charAt(0), arr);
			}
			arr.add(word);
			for (int i = 0; i < word.length(); i++) {
				setDict.add(word.charAt(i));
			}
		}
		for (int i = 0; i < s.length(); i++) {
			if (!setDict.contains(s.charAt(i))) {
				return false;
			}
		}

		Queue<String> q = new LinkedList<String>();
		q.add(s);
		while (!q.isEmpty()) {
			String text = q.remove();
			if (text.isEmpty()) {
				return true;
			}

			char ch = text.charAt(0);
			ArrayList<String> arr = m.get(ch);
			if (arr == null) {
				continue;
			}
			for (String word : arr) {
				if (text.startsWith(word)) {
					String next = text.substring(word.length());
					if (next.isEmpty()) {
						return true;
					}
					q.add(next);
				}
			}
		}
		return false;
	}

	public boolean wordBreakDP(String s, Set<String> dict) {
		if (s == null || s.isEmpty()) {
			return false;
		}

		int len = s.length();
		boolean[] dp = new boolean[len + 1];
		dp[0] = true;
		for (int i = 1; i <= len; i++) {
			for (int j = 0; j < i; j++) {
				if (dp[j] && dict.contains(s.substring(j, i))) {
					dp[i] = true;
					break;
				}
			}
		}

		return dp[len];
	}

	/**
	 * $(Sort List)
	 * 
	 * Sort a linked list in O(n log n) time using constant space complexity.
	 */

	/*
	 * break the whole list into lists of length 1, merge every two of them,
	 * repeat until there is only one list left.
	 */
	public ListNode sortList(ListNode head) {
		ListNode dummy = new ListNode(0), ptr = head;
		dummy.next = head;
		Queue<ListNode> src = new LinkedList<ListNode>(), dst = new LinkedList<ListNode>();
		while (ptr != null) {
			dummy.next = ptr.next;
			ptr.next = null;
			src.offer(ptr);
			ptr = dummy.next;
		}

		while (src.size() > 1) {
			while (!src.isEmpty()) {
				ListNode list1 = src.poll(), list2 = src.poll();
				dst.offer(mergeLists(list1, list2));
			}

			Queue<ListNode> tmp = src;
			src = dst;
			dst = tmp;
		}

		return src.poll();
	}

	private ListNode mergeLists(ListNode l1, ListNode l2) {
		ListNode dummy = new ListNode(0), ptr = dummy;
		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				ptr.next = l1;
				l1 = l1.next;
			} else {
				ptr.next = l2;
				l2 = l2.next;
			}
			ptr = ptr.next;
		}
		if (l1 == null) {
			ptr.next = l2;
		}
		if (l2 == null) {
			ptr.next = l1;
		}

		return dummy.next;
	}

	/**
	 * $(Interleaving String）
	 * 
	 * Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and
	 * s2.
	 * 
	 * For example, Given: s1 = "aabcc", s2 = "dbbca",
	 * 
	 * When s3 = "aadbbcbcac", return true. When s3 = "aadbbbaccc", return
	 * false.
	 */

	public boolean isInterleave(String s1, String s2, String s3) {
		if (s1 == null || s2 == null || s3 == null
				|| s1.length() + s2.length() != s3.length()) {
			return false;
		}

		int len1 = s1.length(), len2 = s2.length();
		boolean[][] dp = new boolean[len1 + 1][len2 + 1];
		dp[0][0] = true;
		for (int col = 1; col <= len2; col++) {
			dp[0][col] = s3.charAt(col - 1) == s2.charAt(col - 1)
					&& dp[0][col - 1];
		}
		for (int row = 1; row <= len1; row++) {
			dp[row][0] = s3.charAt(row - 1) == s1.charAt(row - 1)
					&& dp[row - 1][0];
		}

		for (int row = 1; row <= len1; row++) {
			for (int col = 1; col <= len2; col++) {
				char ch1 = s1.charAt(row - 1);
				char ch2 = s2.charAt(col - 1);
				char ch3 = s3.charAt(row + col - 1);
				dp[row][col] = (ch1 == ch3 && dp[row - 1][col])
						|| (ch2 == ch3 && dp[row][col - 1]);
			}
		}

		return dp[len1][len2];
	}

	/**
	 * $(Minimum Window Substring)
	 * 
	 * Given a string S and a string T, find the minimum window in S which will
	 * contain all the characters in T in complexity O(n).
	 * 
	 * For example, S = "ADOBECODEBANC" T = "ABC" Minimum window is "BANC".
	 * 
	 * Note: If there is no such window in S that covers all characters in T,
	 * return the emtpy string "".
	 * 
	 * If there are multiple such windows, you are guaranteed that there will
	 * always be only one unique minimum window in S.
	 */

	public String minWindow(String S, String T) {
		if (S == null || T == null) {
			return null;
		} else if (S.isEmpty() || T.isEmpty() || S.length() < T.length()) {
			return "";
		}

		int lenS = S.length(), lenT = T.length();
		char[] dict = new char[256], curr = new char[256];
		for (int i = 0; i < lenT; i++) {
			dict[T.charAt(i)]++;
		}

		int cnt = 0, minSize = Integer.MAX_VALUE, start = 0, end = 0;
		for (int i = 0, j = 0; i < lenS; i++) {
			if (cnt < lenT) {
				char ch = S.charAt(i);
				if (curr[ch] < dict[ch]) {
					cnt++;
				}
				curr[ch]++;
			}
			if (cnt == lenT) {
				while (j <= i && curr[S.charAt(j)] > dict[S.charAt(j)]) {
					curr[S.charAt(j)]--;
					j++;
				}
				if (i - j + 1 < minSize) {
					minSize = i - j + 1;
					start = j;
					end = i;
				}
				while (j <= i && cnt == lenT) {
					curr[S.charAt(j)]--;
					if (curr[S.charAt(j)] < dict[S.charAt(j)]) {
						cnt--;
					}
					j++;
				}
			}
		}

		if (minSize == Integer.MAX_VALUE) {
			return "";
		} else {
			return S.substring(start, end + 1);
		}
	}

	public String minWindowTLE(String S, String T) {
		if (S == null || T == null) {
			return null;
		} else if (S.isEmpty() || T.isEmpty() || S.length() < T.length()) {
			return "";
		}

		int lenS = S.length(), lenT = T.length(), minSize = Integer.MAX_VALUE;
		String result = "";
		Map<Character, Integer> dictT = new HashMap<Character, Integer>();
		for (int i = 0; i < lenT; i++) {
			Integer cnt = dictT.get(T.charAt(i));
			if (cnt == null) {
				dictT.put(T.charAt(i), 1);
			} else {
				dictT.put(T.charAt(i), cnt + 1);
			}
		}

		for (int i = 0; i <= lenS - lenT; i++) {
			if (dictT.containsKey(S.charAt(i))) {
				Map<Character, Integer> tempDict = new HashMap<Character, Integer>(
						dictT);
				for (int j = i; j < lenS; j++) {
					Integer cnt = tempDict.get(S.charAt(j));
					if (cnt != null) {
						if (cnt == 1) {
							tempDict.remove(S.charAt(j));
							if (tempDict.isEmpty() && j - i + 1 < minSize) {
								minSize = j - i + 1;
								result = S.substring(i, j + 1);
							}
						} else {
							tempDict.put(S.charAt(j), cnt - 1);
						}
					}
				}
			}
		}

		return result;
	}

	/**
	 * $(Divide Two Integers)
	 * 
	 * Divide two integers without using multiplication, division and mod
	 * operator.
	 */

	public int divide(int dividend, int divisor) {
		long x = (long) dividend, y = (long) divisor;
		boolean negative = (x < 0 && y > 0 || x > 0 && y < 0);
		x = Math.abs(x);
		y = Math.abs(y);
		long ans = 0;
		for (int i = 32; i >= 0; i--) {
			if (x >> i >= y) {
				ans += 1 << i;
				x -= y << i;
			}
		}
		if (negative) {
			ans = ~ans + 1;
		}
		return (int) ans;
	}

	/**
	 * $(Substring with Concatenation of All Words)
	 * 
	 * You are given a string, S, and a list of words, L, that are all of the
	 * same length. Find all starting indices of substring(s) in S that is a
	 * concatenation of each word in L exactly once and without any intervening
	 * characters.
	 * 
	 * For example, given: S: "barfoothefoobarman" L: ["foo", "bar"]
	 * 
	 * You should return the indices: [0,9]. (order does not matter).
	 */

	public ArrayList<Integer> findSubstring(String S, String[] L) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (S == null || S.isEmpty() || L == null || L.length == 0
				|| S.length() < L.length * L[0].length()) {
			return result;
		}

		HashMap<String, Integer> map = new HashMap<String, Integer>();
		for (String word : L) {
			Integer cnt = map.get(word);
			if (cnt == null) {
				map.put(word, 1);
			} else {
				map.put(word, cnt + 1);
			}
		}

		int wlen = L[0].length(), len = L.length * wlen;
		for (int i = 0; i <= S.length() - len; i++) {
			if (isConcatination(S.substring(i, i + len),
					new HashMap<String, Integer>(map), wlen)) {
				result.add(i);
			}
		}

		return result;
	}

	private boolean isConcatination(String S, HashMap<String, Integer> map,
			int wlen) {
		for (int i = 0; i < S.length(); i += wlen) {
			String word = S.substring(i, i + wlen);
			Integer cnt = map.get(word);
			if (cnt == null || cnt <= 0) {
				return false;
			} else {
				map.put(word, cnt - 1);
			}
		}
		return true;
	}

	/**
	 * $(Decode Ways)
	 * 
	 * A message containing letters from A-Z is being encoded to numbers using
	 * the following mapping:
	 * 
	 * 'A' -> 1 'B' -> 2 ... 'Z' -> 26 Given an encoded message containing
	 * digits, determine the total number of ways to decode it.
	 * 
	 * For example, Given encoded message "12", it could be decoded as "AB" (1
	 * 2) or "L" (12).
	 * 
	 * The number of ways decoding "12" is 2.
	 */

	public int numDecodings(String s) {
		if (s == null || s.isEmpty() || s.charAt(0) == '0') {
			return 0;
		}

		int len = s.length();
		int[] dp = new int[len + 1];
		dp[0] = 1;
		dp[1] = 1;
		for (int i = 2; i <= len; i++) {
			if (s.charAt(i - 1) == '0') {
				if (s.charAt(i - 2) < '1' || s.charAt(i - 2) > '2') {
					return 0;
				} else {
					dp[i] = dp[i - 2];
				}
			} else {
				if (s.charAt(i - 2) == '0') {
					dp[i] = dp[i - 2];
				} else if ((s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0' <= 26) {
					dp[i] = dp[i - 1] + dp[i - 2];
				} else {
					dp[i] = dp[i - 1];
				}
			}
		}

		return dp[len];
	}

	/**
	 * $(Word Break II)
	 * 
	 * Given a string s and a dictionary of words dict, add spaces in s to
	 * construct a sentence where each word is a valid dictionary word.
	 * 
	 * Return all such possible sentences.
	 * 
	 * For example, given s = "catsanddog", dict = ["cat", "cats", "and",
	 * "sand", "dog"].
	 * 
	 * A solution is ["cats and dog", "cat sand dog"].
	 */

	public ArrayList<String> wordBreak(String s, Set<String> dict) {
		ArrayList<String> result = new ArrayList<String>();
		if (s == null || s.isEmpty() || dict == null) {
			return result;
		}

		boolean flag = false;
		for (String word : dict) {
			if (s.endsWith(word)) {
				flag = true;
				break;
			}
		}
		if (!flag) {
			return result;
		}

		wordBreakDFS(result, s, "", dict);

		return result;
	}

	private void wordBreakDFS(ArrayList<String> result, String str,
			String sofar, Set<String> dict) {
		if (str.isEmpty()) {
			result.add(sofar);
			return;
		}

		StringBuilder sb = new StringBuilder();
		int len = str.length();
		for (int i = 0; i < len; i++) {
			sb.append(str.charAt(i));
			if (dict.contains(sb.toString())) {
				String space = "";
				if (!sofar.isEmpty()) {
					space = " ";
				}
				wordBreakDFS(result, str.substring(i + 1),
						sofar + space + sb.toString(), dict);
			}
		}
	}

	/**
	 * $(Word Ladder)
	 * 
	 * Given two words (start and end), and a dictionary, find the length of
	 * shortest transformation sequence from start to end, such that:
	 * 
	 * Only one letter can be changed at a time Each intermediate word must
	 * exist in the dictionary For example,
	 * 
	 * Given: start = "hit" end = "cog" dict = ["hot","dot","dog","lot","log"]
	 * As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" ->
	 * "cog", return its length 5.
	 * 
	 * Note: Return 0 if there is no such transformation sequence. All words
	 * have the same length. All words contain only lowercase alphabetic
	 * characters.
	 */

	public int ladderLength(String start, String end, HashSet<String> dict) {
		if (start == null || end == null || start.equals(end) || dict == null) {
			return 0;
		}

		Set<String> source = new HashSet<String>(), dest = new HashSet<String>();
		Set<String> visited = new HashSet<String>();
		source.add(start);
		int count = 1;
		boolean found = false;
		while (!source.isEmpty() && !found) {
			for (String word : source) {
				if (visited.contains(word)) {
					continue;
				}
				visited.add(word);
				if (transform(word, end, dest, visited, dict)) {
					found = true;
					break;
				}
			}

			count++;
			source.clear();
			Set<String> temp = source;
			source = dest;
			dest = temp;
		}

		if (found) {
			return count;
		} else {
			return 0;
		}
	}

	private boolean transform(String word, String end, Set<String> dest,
			Set<String> visited, Set<String> dict) {
		char[] chs = word.toCharArray();
		for (int i = 0; i < chs.length; i++) {
			char ch = chs[i];
			for (char c = 'a'; c <= 'z'; c++) {
				if (ch != c) {
					chs[i] = c;
					String newWord = new String(chs);
					if (dict.contains(newWord) && !visited.contains(newWord)) {
						if (newWord.equals(end)) {
							return true;
						} else {
							dest.add(newWord);
						}
					}
				}
			}
			chs[i] = ch;
		}

		return false;
	}

	/**
	 * $(Word Ladder II)
	 * 
	 * Given two words (start and end), and a dictionary, find all shortest
	 * transformation sequence(s) from start to end, such that:
	 * 
	 * Only one letter can be changed at a time Each intermediate word must
	 * exist in the dictionary For example,
	 * 
	 * Given: start = "hit" end = "cog" dict = ["hot","dot","dog","lot","log"]
	 * Return [ ["hit","hot","dot","dog","cog"], ["hit","hot","lot","log","cog"]
	 * ] Note: All words have the same length. All words contain only lowercase
	 * alphabetic characters.
	 */

	/*
	 * naive BFS, failed to pass large case.
	 */
	public ArrayList<ArrayList<String>> findLaddersTLE(String start,
			String end, HashSet<String> dict) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (start == null || end == null || start.length() != end.length()) {
			return result;
		}
		if (start.equals(end)) {
			ArrayList<String> newPath = new ArrayList<String>();
			newPath.add(start);
			result.add(newPath);
			return result;
		}

		int lenW = start.length();
		Queue<ArrayList<String>> q = new LinkedList<ArrayList<String>>();
		ArrayList<String> arr = new ArrayList<String>();
		arr.add(start);
		q.add(arr);
		boolean found = false;
		int stopCnt = 0;
		HashSet<String> lastHop = new HashSet<String>();

		found: while (!q.isEmpty()) {
			ArrayList<String> path = q.remove();
			if (found && path.size() == stopCnt) {
				break;
			}
			String curr = path.get(path.size() - 1);
			if (found && lastHop.contains(curr)) {
				ArrayList<String> newPath = new ArrayList<String>(path);
				newPath.add(end);
				result.add(newPath);
				continue found;
			}

			dict.remove(curr);

			if (found) {
				int diff = 0;
				for (int i = 0; i < lenW; i++) {
					if (curr.charAt(i) != end.charAt(i)) {
						diff++;
					}
				}
				if (diff > 1) {
					continue found;
				}
			}
			ArrayList<String> trans = new ArrayList<String>();
			for (int i = 0; i < lenW; i++) {
				for (char j = 'a'; j <= 'z'; j++) {
					StringBuilder currsb = new StringBuilder(curr);
					currsb.replace(i, i + 1, String.valueOf(j));
					String next = currsb.toString();
					if (next.equals(end)) {
						found = true;
						lastHop.add(curr);
						stopCnt = path.size() + 1;
						ArrayList<String> newPath = new ArrayList<String>(path);
						newPath.add(end);
						result.add(newPath);
						continue found;
					}
					if (dict.contains(next)) {
						trans.add(next);
					}
				}
			}

			if (!found) {
				for (String next : trans) {
					ArrayList<String> newPath = new ArrayList<String>(path);
					newPath.add(next);
					q.add(newPath);
				}
			}
		}

		return result;
	}

	/*
	 * create a data type to record the word along its path info, not much
	 * faster, TLE.
	 */
	public ArrayList<ArrayList<String>> findLaddersTLE2(String start,
			String end, HashSet<String> dict) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (start == null || end == null || start.length() != end.length()) {
			return result;
		}
		if (start.equals(end)) {
			ArrayList<String> newPath = new ArrayList<String>();
			newPath.add(start);
			result.add(newPath);
			return result;
		}

		int lenW = start.length();
		Queue<PathNode> q = new LinkedList<PathNode>();
		q.add(new PathNode(start, null, 1));
		boolean found = false;
		int pathlen = 0;

		found: while (!q.isEmpty()) {
			PathNode curr = q.remove();
			if (found && curr.layer == pathlen) {
				break;
			}
			dict.remove(curr.word);

			if (found) {
				int diff = 0;
				for (int i = 0; i < lenW; i++) {
					if (curr.word.charAt(i) != end.charAt(i)) {
						diff++;
					}
				}
				if (diff > 1) {
					continue found;
				}
			}
			for (int i = 0; i < lenW; i++) {
				for (char ch = 'a'; ch <= 'z'; ch++) {
					String next = new StringBuilder(curr.word).replace(i,
							i + 1, String.valueOf(ch)).toString();
					if (next.equals(end)) {
						found = true;
						PathNode endNode = new PathNode(end, curr,
								curr.layer + 1);
						pathlen = endNode.layer;
						result.add(retrievePath(endNode));
						continue found;
					}
					if (!found && dict.contains(next)) {
						q.add(new PathNode(next, curr, curr.layer + 1));
					}
				}
			}
		}

		return result;
	}

	/*
	 * use adjacent list: get neighbor of all nodes and save them in a map.
	 */
	public ArrayList<ArrayList<String>> findLaddersAdjList(String start,
			String end, HashSet<String> dict) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (start == null || end == null || start.length() != end.length()) {
			return result;
		}
		if (start.equals(end)) {
			ArrayList<String> newPath = new ArrayList<String>();
			newPath.add(start);
			result.add(newPath);
			return result;
		}

		int lenW = start.length();
		HashMap<String, HashSet<String>> nbrs = new HashMap<String, HashSet<String>>();
		for (String word : dict) {
			HashSet<String> nbr = new HashSet<String>();
			char[] chArr = word.toCharArray();
			for (int i = 0; i < lenW; i++) {
				char orig = chArr[i];
				for (char ch = 'a'; ch <= 'z'; ch++) {
					chArr[i] = ch;
					String next = new String(chArr);
					if (dict.contains(next)) {
						nbr.add(next);
					}
				}
				chArr[i] = orig;
			}
			nbr.remove(word);
			nbrs.put(word, nbr);
		}

		Queue<PathNode> q = new LinkedList<PathNode>();
		q.add(new PathNode(start, null, 1));
		boolean found = false;
		int pathlen = 0;
		found: while (!q.isEmpty()) {
			PathNode curr = q.remove();
			if (found && curr.layer == pathlen) {
				break;
			}
			dict.remove(curr.word);

			if (found) {
				int diff = 0;
				for (int i = 0; i < lenW; i++) {
					if (curr.word.charAt(i) != end.charAt(i)) {
						diff++;
					}
				}
				if (diff > 1) {
					continue found;
				}
			}
			HashSet<String> nbr = nbrs.get(curr.word);
			for (String next : nbr) {
				if (next.equals(end)) {
					found = true;
					PathNode endNode = new PathNode(end, curr, curr.layer + 1);
					pathlen = endNode.layer;
					result.add(retrievePath(endNode));
					continue found;
				}
				if (!found && dict.contains(next)) {
					q.add(new PathNode(next, curr, curr.layer + 1));
				}
			}

		}

		return result;
	}

	/*
	 * some improvement
	 */
	public ArrayList<ArrayList<String>> findLaddersAdjList2(String start,
			String end, HashSet<String> dict) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (start == null || end == null || start.length() != end.length()) {
			return result;
		}
		if (start.equals(end)) {
			ArrayList<String> newPath = new ArrayList<String>();
			newPath.add(start);
			result.add(newPath);
			return result;
		}

		int lenW = start.length();
		HashMap<String, HashSet<String>> nbrs = new HashMap<String, HashSet<String>>();
		for (String word : dict) {
			HashSet<String> nbr = new HashSet<String>();
			char[] chArr = word.toCharArray();
			for (int i = 0; i < lenW; i++) {
				char orig = chArr[i];
				for (char ch = 'a'; ch <= 'z'; ch++) {
					chArr[i] = ch;
					String next = new String(chArr);
					if (dict.contains(next)) {
						nbr.add(next);
					}
				}
				chArr[i] = orig;
			}
			nbr.remove(word);
			nbrs.put(word, nbr);
		}

		Queue<PathNode> q = new LinkedList<PathNode>();
		q.add(new PathNode(start, null, 1));
		boolean found = false;
		int pathlen = 0;
		found: while (!q.isEmpty()) {
			PathNode curr = q.remove();
			if (found && curr.layer == pathlen + 1) {
				break found;
			}
			if (curr.word.equals(end)) {
				found = true;
				pathlen = curr.layer;
				result.add(retrievePath(curr));
				continue found;
			}

			dict.remove(curr.word);
			HashSet<String> nbr = nbrs.get(curr.word);
			ArrayList<String> toRmv = new ArrayList<String>();
			for (String next : nbr) {
				if (!found) {
					if (dict.contains(next)) {
						q.add(new PathNode(next, curr, curr.layer + 1));
					} else {
						toRmv.add(next);
					}
				}
			}
			for (String word : toRmv) {
				nbr.remove(word);
			}
		}

		return result;
	}

	private ArrayList<String> retrievePath(PathNode node) {
		Stack<String> s = new Stack<String>();
		while (node != null) {
			s.push(node.word);
			node = node.prev;
		}
		ArrayList<String> path = new ArrayList<String>();
		while (!s.isEmpty()) {
			path.add(s.pop());
		}
		return path;
	}

	class PathNode {
		public String word;
		public PathNode prev;
		public int layer;

		public PathNode(String word, PathNode prev, int layer) {
			this.word = word;
			this.prev = prev;
			this.layer = layer;
		}
	}

	public ArrayList<ArrayList<String>> findLadders(String start, String end,
			HashSet<String> dict) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (start == null || end == null || dict == null) {
			return result;
		} else if (start.equals(end)) {
			ArrayList<String> sequence = new ArrayList<String>();
			sequence.add(start);
			result.add(sequence);
		}

		Set<String> src = new HashSet<String>(), dst = new HashSet<String>(), vst = new HashSet<String>();
		Map<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
		src.add(start);
		boolean found = false;
		int count = 1;
		while (!src.isEmpty() && !found) {
			for (String curr : src) {
				if (vst.contains(curr)) {
					continue;
				}
				vst.add(curr);
				if (trans(curr, end, dst, vst, dict, map)) {
					found = true;
				}
			}

			count++;
			src.clear();
			Set<String> tmp = src;
			src = dst;
			dst = tmp;
		}

		if (found) {
			ArrayList<String> sequence = new ArrayList<String>();
			sequence.add(end);
			generatedSequences(result, sequence, count, map);
		}
		return result;
	}

	private boolean trans(String curr, String end, Set<String> dst,
			Set<String> vst, Set<String> dict,
			Map<String, ArrayList<String>> map) {
		char[] chs = curr.toCharArray();
		boolean flag = false;
		for (int i = 0; i < chs.length; i++) {
			char ch = chs[i];
			for (char c = 'a'; c <= 'z'; c++) {
				if (ch != c) {
					chs[i] = c;
					String next = new String(chs);
					if (dict.contains(next) && !vst.contains(next)) {
						if (end.equals(next)) {
							flag = true;
						} else {
							dst.add(next);
						}
						ArrayList<String> list = map.get(next);
						if (list == null) {
							list = new ArrayList<String>();
							map.put(next, list);
						}
						list.add(curr);
						if (flag) {
							return true;
						}
					}
				}
			}
			chs[i] = ch;
		}

		return false;
	}

	private void generatedSequences(ArrayList<ArrayList<String>> result,
			ArrayList<String> sequence, int count,
			Map<String, ArrayList<String>> map) {
		if (count <= 0) {
			return;
		}

		String curr = sequence.get(sequence.size() - 1);
		if (count == 1 && map.get(curr) == null) {
			ArrayList<String> seq = new ArrayList<String>();
			for (int i = sequence.size() - 1; i >= 0; i--) {
				seq.add(sequence.get(i));
			}
			result.add(seq);
			return;
		}

		for (String next : map.get(curr)) {
			sequence.add(next);
			generatedSequences(result, sequence, count - 1, map);
			sequence.remove(sequence.size() - 1);
		}
	}

	@Test
	public void testFindLadder() {
		HashSet<String> dict = new HashSet<String>();
		String[] words = { "dose", "ends", "dine", "jars", "prow", "soap",
				"guns", "hops", "cray", "hove", "ella", "hour", "lens", "jive",
				"wiry", "earl", "mara", "part", "flue", "putt", "rory", "bull",
				"york", "ruts", "lily", "vamp", "bask", "peer", "boat", "dens",
				"lyre", "jets", "wide", "rile", "boos", "down", "path", "onyx",
				"mows", "toke", "soto", "dork", "nape", "mans", "loin", "jots",
				"male", "sits", "minn", "sale", "pets", "hugo", "woke", "suds",
				"rugs", "vole", "warp", "mite", "pews", "lips", "pals", "nigh",
				"sulk", "vice", "clod", "iowa", "gibe", "shad", "carl", "huns",
				"coot", "sera", "mils", "rose", "orly", "ford", "void", "time",
				"eloy", "risk", "veep", "reps", "dolt", "hens", "tray", "melt",
				"rung", "rich", "saga", "lust", "yews", "rode", "many", "cods",
				"rape", "last", "tile", "nosy", "take", "nope", "toni", "bank",
				"jock", "jody", "diss", "nips", "bake", "lima", "wore", "kins",
				"cult", "hart", "wuss", "tale", "sing", "lake", "bogy", "wigs",
				"kari", "magi", "bass", "pent", "tost", "fops", "bags", "duns",
				"will", "tart", "drug", "gale", "mold", "disk", "spay", "hows",
				"naps", "puss", "gina", "kara", "zorn", "boll", "cams", "boas",
				"rave", "sets", "lego", "hays", "judy", "chap", "live", "bahs",
				"ohio", "nibs", "cuts", "pups", "data", "kate", "rump", "hews",
				"mary", "stow", "fang", "bolt", "rues", "mesh", "mice", "rise",
				"rant", "dune", "jell", "laws", "jove", "bode", "sung", "nils",
				"vila", "mode", "hued", "cell", "fies", "swat", "wags", "nate",
				"wist", "honk", "goth", "told", "oise", "wail", "tels", "sore",
				"hunk", "mate", "luke", "tore", "bond", "bast", "vows", "ripe",
				"fond", "benz", "firs", "zeds", "wary", "baas", "wins", "pair",
				"tags", "cost", "woes", "buns", "lend", "bops", "code", "eddy",
				"siva", "oops", "toed", "bale", "hutu", "jolt", "rife", "darn",
				"tape", "bold", "cope", "cake", "wisp", "vats", "wave", "hems",
				"bill", "cord", "pert", "type", "kroc", "ucla", "albs", "yoko",
				"silt", "pock", "drub", "puny", "fads", "mull", "pray", "mole",
				"talc", "east", "slay", "jamb", "mill", "dung", "jack", "lynx",
				"nome", "leos", "lade", "sana", "tike", "cali", "toge", "pled",
				"mile", "mass", "leon", "sloe", "lube", "kans", "cory", "burs",
				"race", "toss", "mild", "tops", "maze", "city", "sadr", "bays",
				"poet", "volt", "laze", "gold", "zuni", "shea", "gags", "fist",
				"ping", "pope", "cora", "yaks", "cosy", "foci", "plan", "colo",
				"hume", "yowl", "craw", "pied", "toga", "lobs", "love", "lode",
				"duds", "bled", "juts", "gabs", "fink", "rock", "pant", "wipe",
				"pele", "suez", "nina", "ring", "okra", "warm", "lyle", "gape",
				"bead", "lead", "jane", "oink", "ware", "zibo", "inns", "mope",
				"hang", "made", "fobs", "gamy", "fort", "peak", "gill", "dino",
				"dina", "tier" };
		for (String word : words) {
			dict.add(word);
		}
		System.out.println(dict.size());
		ArrayList<ArrayList<String>> result = findLadders("cali", "fobs", dict);
		System.out.println(result.size());
		for (ArrayList<String> path : result) {
			System.out.println(path);
		}
		System.out.println(dict.size());
	}

	/**
	 * $(Regular Expression Matching)
	 * 
	 * Implement regular expression matching with support for '.' and '*'.
	 * 
	 * '.' Matches any single character. '*' Matches zero or more of the
	 * preceding element.
	 * 
	 * The matching should cover the entire input string (not partial).
	 * 
	 * The function prototype should be: bool isMatch(const char *s, const char
	 * *p)
	 * 
	 * Some examples: isMatch("aa","a") → false isMatch("aa","aa") → true
	 * isMatch("aaa","aa") → false isMatch("aa", "a*") → true isMatch("aa",
	 * ".*") → true isMatch("ab", ".*") → true isMatch("aab", "c*a*b") → true
	 */

	public boolean isMatchRegExp(String s, String p) {
		if (s == null || p == null) {
			return false;
		}

		if (p.isEmpty()) {
			return s.isEmpty();
		} else if (s.isEmpty()) {
			if (p.length() >= 2 && p.charAt(0) != '*' && p.charAt(1) == '*') {
				return isMatch(s, p.substring(2));
			} else {
				return false;
			}
		}

		char lasts = s.charAt(s.length() - 1), lastp = p.charAt(p.length() - 1);
		if (lastp != '*' && lastp != '.' && lastp != lasts) {
			return false;
		}

		if (p.charAt(0) == '*') {
			return false;
		}

		if (p.length() > 1 && p.charAt(1) == '*') {
			if (p.charAt(0) == '.') {
				for (int i = s.length(); i >= 0; i--) {
					if (isMatch(s.substring(i), p.substring(2))) {
						return true;
					}
				}
				return false;
			} else {
				int index = 0;
				while (index < s.length() && s.charAt(index) == p.charAt(0)) {
					index++;
				}
				for (int i = index; i >= 0; i--) {
					if (isMatch(s.substring(i), p.substring(2))) {
						return true;
					}
				}
				return false;
			}
		} else if (p.charAt(0) == '.') {
			return isMatch(s.substring(1), p.substring(1));
		} else {
			return p.charAt(0) == s.charAt(0)
					&& isMatch(s.substring(1), p.substring(1));
		}
	}

	/**
	 * $(Wildcard Matching)
	 * 
	 * Implement wildcard pattern matching with support for '?' and '*'.
	 * 
	 * '?' Matches any single character. '*' Matches any sequence of characters
	 * (including the empty sequence).
	 * 
	 * The matching should cover the entire input string (not partial).
	 * 
	 * The function prototype should be: bool isMatch(const char *s, const char
	 * *p)
	 * 
	 * Some examples: isMatch("aa","a") → false isMatch("aa","aa") → true
	 * isMatch("aaa","aa") → false isMatch("aa", "*") → true isMatch("aa", "a*")
	 * → true isMatch("ab", "?*") → true isMatch("aab", "c*a*b") → false
	 */

	/*
	 * DFS, TLE
	 */
	public boolean isMatchDFS(String s, String p) {
		if (s == null || p == null) {
			return false;
		}

		if (s.isEmpty()) {
			for (int i = 0; i < p.length(); i++) {
				if (p.charAt(i) != '*') {
					return false;
				}
			}
			return true;
		} else if (p.isEmpty()) {
			return s.isEmpty();
		}

		char lastp = p.charAt(p.length() - 1);
		if (lastp != '*' && lastp != '?' && lastp != s.charAt(s.length() - 1)) {
			return false;
		}

		if (p.charAt(0) == '?') {
			return isMatchDFS(s.substring(1), p.substring(1));
		} else if (p.charAt(0) == '*') {
			int end = 0;
			while (end < p.length() && p.charAt(end) == '*') {
				end++;
			}
			for (int i = s.length(); i >= 0; i--) {
				if (isMatchDFS(s.substring(i), p.substring(end))) {
					return true;
				}
			}
			return false;
		} else {
			if (s.charAt(0) == p.charAt(0)) {
				return isMatchDFS(s.substring(1), p.substring(1));
			} else {
				return false;
			}
		}
	}

	/*
	 * DP, lots of optimizations needed: 1. remove consecutive * from p; 2.
	 * check empty s and empty p at the start; 3. check if p has more non-*
	 * characters than s;
	 */
	public boolean isMatchDP(String s, String p) {
		int lenS = s.length(), lenP = p.length();
		if (lenP == 0) {
			return lenS == 0;
		}

		if (lenS == 0) {
			for (int i = 0; i < lenP; i++) {
				if (p.charAt(i) != '*') {
					return false;
				}
			}
			return true;
		}

		if (p.charAt(0) != '*' && p.charAt(0) != '?'
				&& p.charAt(0) != s.charAt(0)) {
			return false;
		}
		if (p.charAt(lenP - 1) != '*' && p.charAt(lenP - 1) != '?'
				&& p.charAt(lenP - 1) != s.charAt(lenS - 1)) {
			return false;
		}

		char[] chs = p.toCharArray();
		int index = 0, cnt = 0;
		for (int i = 0; i < lenP; i++) {
			chs[index++] = chs[i];
			if (chs[i] == '*') {
				while (i < lenP - 1 && chs[i + 1] == '*') {
					i++;
				}
			} else {
				cnt++;
			}
		}
		if (cnt > lenS) {
			return false;
		}
		p = new String(chs, 0, index);
		lenP = p.length();

		int w = lenS + 1;
		boolean[][] dp = new boolean[2][w];
		if (p.charAt(0) == '*') {
			for (int c = 0; c < w; c++) {
				dp[0][c] = true;
			}
		} else {
			dp[0][1] = p.charAt(0) == '?' || p.charAt(0) == s.charAt(0);
		}
		boolean hasMatch = dp[0][1];
		for (int r = 1; r < lenP; r++) {
			int row = r & 1;
			dp[row][0] = false;
			int lastRow = (r - 1) & 1;
			for (int c = 1; c < w; c++) {
				char ch = p.charAt(r);
				if (ch == '*') {
					boolean found = false;
					for (int col = 1; col <= c; col++) {
						if (dp[lastRow][col]) {
							found = true;
							break;
						}
					}
					dp[row][c] = found;
				} else {
					dp[row][c] = (ch == '?' || ch == s.charAt(c - 1))
							&& dp[lastRow][c - 1];
				}
				if (dp[row][c]) {
					hasMatch = true;
				}
			}
			if (!hasMatch) {
				return false;
			}
			hasMatch = false;
		}

		return dp[(lenP - 1) & 1][lenS];
	}

	/*
	 * O(n^2) time, O(1) space, optimal
	 */
	public boolean isMatch(String s, String p) {
		if (s == null || p == null) {
			return false;
		}

		int lens = s.length(), lenp = p.length();
		int ps = 0, pp = 0, star = -1, sstar = ps;
		while (ps < lens) {
			if (pp != lenp
					&& (p.charAt(pp) == '?' || s.charAt(ps) == p.charAt(pp))) {
				ps++;
				pp++;
			} else if (pp != lenp && p.charAt(pp) == '*') {
				star = pp;
				pp++;
				sstar = ps;
			} else {
				if (star != -1) {
					pp = star + 1;
					sstar++;
					ps = sstar;
				} else {
					return false;
				}
			}
		}

		while (pp < lenp && p.charAt(pp) == '*') {
			pp++;
		}
		return pp == lenp;
	}

	/**
	 * $(String to Integer (atoi))
	 * 
	 * Implement atoi to convert a string to an integer.
	 * 
	 * Hint: Carefully consider all possible input cases. If you want a
	 * challenge, please do not see below and ask yourself what are the possible
	 * input cases.
	 * 
	 * Notes: It is intended for this problem to be specified vaguely (ie, no
	 * given input specs). You are responsible to gather all the input
	 * requirements up front.
	 * 
	 * Requirements for atoi: The function first discards as many whitespace
	 * characters as necessary until the first non-whitespace character is
	 * found. Then, starting from this character, takes an optional initial plus
	 * or minus sign followed by as many numerical digits as possible, and
	 * interprets them as a numerical value.
	 * 
	 * The string can contain additional characters after those that form the
	 * integral number, which are ignored and have no effect on the behavior of
	 * this function.
	 * 
	 * If the first sequence of non-whitespace characters in str is not a valid
	 * integral number, or if no such sequence exists because either str is
	 * empty or it contains only whitespace characters, no conversion is
	 * performed.
	 * 
	 * If no valid conversion could be performed, a zero value is returned. If
	 * the correct value is out of the range of representable values, INT_MAX
	 * (2147483647) or INT_MIN (-2147483648) is returned.
	 */

	public int atoi(String str) {
		if (str == null) {
			return 0;
		}

		str = str.trim();
		if (str.isEmpty()) {
			return 0;
		}

		String sign = "";
		int start = 0;
		int maxVal;
		if (str.charAt(0) == '-') {
			sign = "-";
			start = 1;
			maxVal = Integer.MIN_VALUE;
		} else if (str.charAt(0) == '+') {
			start = 1;
			maxVal = Integer.MAX_VALUE;
		} else if (Character.isDigit(str.charAt(0))) {
			maxVal = Integer.MAX_VALUE;
		} else {
			return 0;
		}
		int i = start;
		for (; i < str.length(); i++) {
			if (!Character.isDigit(str.charAt(i))) {
				break;
			} else if (i - start + 1 > 10) {
				return maxVal;
			}
		}
		String num = str.substring(start, i);
		if (num.isEmpty()) {
			return 0;
		} else {
			long l = Long.parseLong(sign + num);
			if (l < Integer.MIN_VALUE) {
				return Integer.MIN_VALUE;
			} else if (l > Integer.MAX_VALUE) {
				return Integer.MAX_VALUE;
			} else {
				return Integer.parseInt(sign + num);
			}
		}
	}

	/**
	 * $(Valid Number)
	 * 
	 * Validate if a given string is numeric.
	 * 
	 * Some examples: "0" => true " 0.1 " => true "abc" => false "1 a" => false
	 * "2e10" => true Note: It is intended for the problem statement to be
	 * ambiguous. You should gather all requirements up front before
	 * implementing one.
	 */

	public boolean isNumberRegExp(String s) {
		return s != null
				&& (s.trim()
						.matches("[+-]?(\\d+|\\d+\\.(\\d+)?|(\\d+)?\\.\\d+)([eE][+-]?\\d+)?"));
	}

	public boolean isNumber(String s) {
		if (s == null || s.isEmpty()) {
			return false;
		}

		int l = 0, len = s.length();
		while (l < len && Character.isWhitespace(s.charAt(l))) {
			l++;
		}
		if (l == len) {
			return false;
		}
		int r = len - 1;
		while (r >= 0 && Character.isWhitespace(s.charAt(r))) {
			r--;
		}
		if (l == r) {
			return Character.isDigit(s.charAt(l));
		}

		s = s.substring(l, r + 1);
		len = s.length();
		HashMap<String, ArrayList<Integer>> m = new HashMap<String, ArrayList<Integer>>();
		m.put("+-", new ArrayList<Integer>());
		m.put("dot", new ArrayList<Integer>());
		m.put("eE", new ArrayList<Integer>());
		// d. .d d.d d
		for (int i = 0; i < len; i++) {
			char ch = s.charAt(i);
			if (ch == '-' || ch == '+') {
				ArrayList<Integer> arr = m.get("+-");
				if (arr.size() == 2) {
					return false;
				}
				arr.add(i);
			} else if (ch == 'e' || ch == 'E') {
				ArrayList<Integer> arr = m.get("eE");
				if (arr.size() == 1) {
					return false;
				}
				arr.add(i);
			} else if (ch == '.') {
				ArrayList<Integer> arr = m.get("dot");
				if (arr.size() == 1) {
					return false;
				}
				arr.add(i);
			} else if (!Character.isDigit(ch)) {
				return false;
			}
		}

		// +- dot eE +-
		// 1. eE and second +- has to occur or absent together, 8 cases
		// eliminated
		int start = 0, end = len - 1;
		int[] signs = new int[4];
		Arrays.fill(signs, -1);
		ArrayList<Integer> arr = m.get("eE");
		if (arr.size() > 0) {
			signs[2] = arr.get(0);
		}
		arr = m.get("dot");
		if (arr.size() > 0) {
			signs[1] = arr.get(0);
		}
		if (signs[2] != -1 && signs[1] > signs[2]) {
			return false;
		}

		arr = m.get("+-");
		if (arr.size() == 2) {
			if (arr.get(0) != start || signs[2] == -1
					|| arr.get(1) != signs[2] + 1) {
				return false;
			}
			if (signs[1] == -1 && signs[2] - arr.get(0) == 1 || signs[1] != -1
					&& signs[2] - arr.get(0) == 2) {
				return false;
			}
			return end > arr.get(1);
		} else if (arr.size() == 1) {
			if (signs[2] == -1) {
				return arr.get(0) == start && len > (signs[1] == -1 ? 1 : 2);
			} else {
				if (arr.get(0) == start) {
					return signs[2] - start > (signs[1] == -1 ? 1 : 2)
							&& end > signs[2];
				} else if (arr.get(0) == signs[2] + 1) {
					return signs[2] - start > (signs[1] == -1 ? 0 : 1)
							&& end > arr.get(0);
				} else {
					return false;
				}
			}
		} else {
			if (signs[1] != -1 && signs[2] == -1) {
				return len >= 2;
			} else if (signs[1] == -1 && signs[2] != -1) {
				return signs[2] > start && signs[2] < end;
			} else if (signs[1] != -1 && signs[2] != -1) {
				return signs[2] - start > 1 && signs[2] < end;
			} else {
				return true;
			}
		}
	}

	@Test
	public void testIsNumber() {
		String[] nums = { " ", " 1", " 1 ", "a", " a ", "1a", "1.", ".1",
				"1.1", "1.e", ".1e", "1.1e", "1.1e1", "1.1e1.", "1.1e1.1",
				"e1", "-.1e1", "+1.e1", "+1.1e1", "+1.1e", "-1.1e1.1",
				"-123.4ae1", " +2e1q23  ", " + 1e12  ", " + 123 e -1",
				"+13e-1   ", " +123 ", "-33" };
		// String[] nums = {"-.1e1", "+1.e1", "+1.1e1",};
		for (String num : nums) {
			System.out.println(num + " " + isNumber(num));
		}
	}

	/**
	 * $(Text Justification)
	 * 
	 * Given an array of words and a length L, format the text such that each
	 * line has exactly L characters and is fully (left and right) justified.
	 * 
	 * You should pack your words in a greedy approach; that is, pack as many
	 * words as you can in each line. Pad extra spaces ' ' when necessary so
	 * that each line has exactly L characters.
	 * 
	 * Extra spaces between words should be distributed as evenly as possible.
	 * If the number of spaces on a line do not divide evenly between words, the
	 * empty slots on the left will be assigned more spaces than the slots on
	 * the right.
	 * 
	 * For the last line of text, it should be left justified and no extra space
	 * is inserted between words.
	 * 
	 * For example, words: ["This", "is", "an", "example", "of", "text",
	 * "justification."] L: 16.
	 * 
	 * Return the formatted lines as: [ "This    is    an", "example  of text",
	 * "justification.  " ] Note: Each word is guaranteed not to exceed L in
	 * length.
	 * 
	 * Corner Cases: A line other than the last line might contain only one
	 * word. What should you do in this case? In this case, that line should be
	 * left-justified.
	 */

	public ArrayList<String> fullJustify(String[] words, int L) {
		ArrayList<String> result = new ArrayList<String>();
		if (words == null || words.length == 0) {
			return result;
		}

		int start = 0, end = 1;
		boolean stop = false;
		;
		while (!stop) {
			int lenSum = words[start].length(), cnt = 1;
			while (end < words.length && words[end].length() + lenSum + 1 <= L) {
				lenSum += words[end].length() + 1;
				cnt++;
				end++;
			}
			int[] spaceLen = new int[cnt];
			if (end == words.length) {
				stop = true;
				Arrays.fill(spaceLen, 1);
				spaceLen[spaceLen.length - 1] = L - lenSum;
			} else if (cnt == 1) {
				spaceLen[0] = L - lenSum;
			} else {
				int remain = L - lenSum, space = remain / (cnt - 1) + 1, mod = remain
						% (cnt - 1);
				Arrays.fill(spaceLen, space);
				for (int i = 0; i < mod; i++) {
					spaceLen[i]++;
				}
				spaceLen[spaceLen.length - 1] = 0;
			}
			StringBuilder sb = new StringBuilder();
			for (int i = start; i < end; i++) {
				sb.append(words[i]);
				for (int k = 0; k < spaceLen[i - start]; k++) {
					sb.append(" ");
				}
			}
			result.add(sb.toString());

			start = end;
			end = start + 1;
		}

		return result;
	}

	/**
	 * $(Median of Two Sorted Arrays)
	 * 
	 * There are two sorted arrays A and B of size m and n respectively. Find
	 * the median of the two sorted arrays. The overall run time complexity
	 * should be O(log (m+n)).
	 */

	public double findMedianSortedArrays(int A[], int B[]) {
		int lenA = A.length, lenB = B.length;
		if (((lenA + lenB) & 1) == 0) {
			return (findNth(A, 0, B, 0, (lenA + lenB) / 2 + 1) + findNth(A, 0,
					B, 0, (lenA + lenB) / 2)) / 2.0;
		} else {
			return findNth(A, 0, B, 0, (lenA + lenB) / 2 + 1);
		}
	}

	private int findNth(int[] A, int startA, int[] B, int startB, int n) {
		int lenA = A.length, lenB = B.length;
		if (lenA > lenB) {
			return findNth(B, startB, A, startA, n);
		}

		if (startA == lenA) {
			return B[startB + n - 1];
		} else if (startB == lenB) {
			return A[startA + n - 1];
		} else if (n == 1) {
			return Math.min(A[startA], B[startB]);
		}

		int partA = Math.min(n / 2, lenA - startA), partB = n - partA;
		int nthA = A[startA + partA - 1], nthB = B[startB + partB - 1];
		if (nthA == nthB) {
			return nthA;
		} else if (nthA > nthB) {
			return findNth(A, startA, B, startB + partB, n - partB);
		} else {
			return findNth(A, startA + partA, B, startB, n - partA);
		}
	}

	/**
	 * $(LRU Cache)
	 * 
	 * Design and implement a data structure for Least Recently Used (LRU)
	 * cache. It should support the following operations: get and set.
	 * 
	 * get(key) - Get the value (will always be positive) of the key if the key
	 * exists in the cache, otherwise return -1. set(key, value) - Set or insert
	 * the value if the key is not already present. When the cache reached its
	 * capacity, it should invalidate the least recently used item before
	 * inserting a new item.
	 */

	public class LRUCache {
		private class LRUNode {
			private int key;
			private int val;
			private LRUNode prev;
			private LRUNode next;

			public LRUNode(int key, int val) {
				this.key = key;
				this.val = val;
				prev = null;
				next = null;
			}
		}

		private LRUNode head;
		private LRUNode tail;
		private Map<Integer, LRUNode> map;
		private final int capacity;
		private int size = 0;

		public LRUCache(int capacity) {
			this.capacity = capacity;
			this.head = new LRUNode(0, 0);
			this.tail = new LRUNode(0, 0);
			head.next = tail;
			tail.prev = head;
			this.map = new HashMap<Integer, LRUNode>();
		}

		public int get(int key) {
			LRUNode node = map.get(key);
			if (node == null) {
				return -1;
			} else {
				remove(node);
				insert(node);
				return node.val;
			}
		}

		public void set(int key, int value) {
			LRUNode node = map.get(key);
			if (node == null) {
				node = new LRUNode(key, value);
				map.put(key, node);
				insert(node);
				size++;
				if (size > capacity) {
					map.remove(tail.prev.key);
					remove(tail.prev);
					size--;
				}
			} else {
				node.val = value;
				remove(node);
				insert(node);
			}
		}

		private void remove(LRUNode node) {
			node.prev.next = node.next;
			node.next.prev = node.prev;
		}

		private void insert(LRUNode node) {
			node.next = head.next;
			node.prev = head;
			head.next = node;
			node.next.prev = node;
		}
	}

	/**
	 * $(Max Points on a Line)
	 * 
	 * Given n points on a 2D plane, find the maximum number of points that lie
	 * on the same straight line.
	 */

	class Point {
		int x;
		int y;

		Point() {
			x = 0;
			y = 0;
		}

		Point(int a, int b) {
			x = a;
			y = b;
		}
	}

	public int maxPoints(Point[] points) {
		if (points == null || points.length == 0) {
			return 0;
		}

		Arrays.sort(points, new PointsCmp());
		int max = 1;
		for (int i = 0; i < points.length - 1; i++) {
			int startCnt = 1;
			Point p1 = points[i];
			for (int j = i + 1; j < points.length; j++) {
				Point p2 = points[j];
				if (p1.x == p2.x && p1.y == p2.y) {
					startCnt++;
					max = Math.max(max, startCnt);
				} else {
					int endCnt = 1;
					for (int k = j + 1; k < points.length; k++) {
						Point p3 = points[k];
						if ((p3.y - p1.y) * (p2.x - p1.x) == (p2.y - p1.y)
								* (p3.x - p1.x)) {
							endCnt++;
						}
					}
					max = Math.max(max, startCnt + endCnt);
				}
			}
		}

		return max;
	}

	private class PointsCmp implements Comparator<Point> {
		public int compare(Point p1, Point p2) {
			if (p1.x != p2.x) {
				return p1.x - p2.x;
			} else {
				return p1.y - p2.y;
			}
		}
	}

	/**
	 * $(Reverse Words in a String)
	 * 
	 * Given an input string, reverse the string word by word.
	 * 
	 * For example, Given s = "the sky is blue", return "blue is sky the".
	 * 
	 * Clarification: What constitutes a word? A sequence of non-space
	 * characters constitutes a word. Could the input string contain leading or
	 * trailing spaces? Yes. However, your reversed string should not contain
	 * leading or trailing spaces. How about multiple spaces between two words?
	 * Reduce them to a single space in the reversed string.
	 */

	public String reverseWords(String s) {
		if (s == null || s.isEmpty()) {
			return s;
		}

		char[] chs = s.toCharArray();
		int left = 0, right = chs.length - 1;
		while (left < chs.length && chs[left] == ' ') {
			left++;
		}
		while (right >= left && chs[right] == ' ') {
			right--;
		}

		if (right <= left) {
			return s.substring(left, right + 1);
		}

		int bp = 0, fp = left;
		boolean flag = true;
		for (; fp <= right; fp++) {
			if (chs[fp] != ' ') {
				chs[bp++] = chs[fp];
				flag = true;
			} else {
				if (flag) {
					chs[bp++] = chs[fp];
					flag = false;
				}
			}
		}

		int start = 0, end = 0;
		while (start < bp) {
			while (end < bp && chs[end] != ' ') {
				end++;
			}
			reverseString(chs, start, end - 1);
			start = end + 1;
			end = start;
		}

		reverseString(chs, 0, bp - 1);
		return new String(chs, 0, bp);
	}

	private void reverseString(char[] chs, int start, int end) {
		while (start < end) {
			chs[start] ^= chs[end];
			chs[end] ^= chs[start];
			chs[start] ^= chs[end];
			start++;
			end--;
		}
	}

	/**
	 * $(Implement strStr())
	 * 
	 * Implement strStr().
	 * 
	 * Returns a pointer to the first occurrence of needle in haystack, or null
	 * if needle is not part of haystack.
	 */

	public String strStr(String haystack, String needle) {
		if (haystack == null || needle == null
				|| needle.length() > haystack.length()) {
			return null;
		}

		int len1 = haystack.length(), len2 = needle.length();
		int i, j;
		for (i = 0; i < len1 - len2 + 1; i++) {
			for (j = 0; j < len2; j++) {
				if (needle.charAt(j) != haystack.charAt(i + j)) {
					break;
				}
			}
			if (j == len2) {
				return haystack.substring(i);
			}
		}

		return null;
	}
}