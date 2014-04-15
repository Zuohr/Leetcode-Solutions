import java.util.Random;

import org.junit.Test;

public class AddList {
	class ListNode {
		public int val;
		public ListNode next;

		public ListNode(int val) {
			this.val = val;
		}
	}

	public ListNode addList(ListNode l1, ListNode l2) {
		if (l1 == null) {
			return l2;
		} else if (l2 == null) {
			return l1;
		}

		int len1 = 0, len2 = 0;
		ListNode ptr = l1;
		while (ptr != null) {
			len1++;
			ptr = ptr.next;
		}
		ptr = l2;
		while (ptr != null) {
			len2++;
			ptr = ptr.next;
		}

		if (len1 < len2) {
			ListNode temp = l1;
			l1 = l2;
			l2 = temp;
			len1 ^= len2;
			len2 ^= len1;
			len1 ^= len2;
		}
		

		for (int i = 0; i < len1 - len2; i++) {
			ListNode pad = new ListNode(0);
			pad.next = l2;
			l2 = pad;
		}

		ListNode result = addListRecursion(l1, l2);
		if (result.val >= 10) {
			ListNode newHead = new ListNode(1);
			newHead.next = result;
			result = newHead;
			result.val %= 10;
		}
		return result;
	}

	private ListNode addListRecursion(ListNode l1, ListNode l2) {
		if (l1 == null || l2 == null) {
			return null;
		}

		ListNode head = addListRecursion(l1.next, l2.next);
		int carry = 0;
		if (head != null) {
			carry = head.val / 10;
			head.val %= 10;
		}
		ListNode newHead = new ListNode(l1.val + l2.val + carry);
		newHead.next = head;
		return newHead;
	}

	private void printList(ListNode head) {
		while (head != null) {
			System.out.print(head.val + " -> ");
			head = head.next;
		}
		System.out.println("null");
	}

	@Test
	public void test() {
		Random r = new Random();
		ListNode dummy1 = new ListNode(0), ptr1 = dummy1;
		int len1 = 2, len2 = 5;
		for (int i = 0; i < len1; i++) {
			ptr1.next = new ListNode(r.nextInt(10));
			ptr1 = ptr1.next;
		}

		ListNode dummy2 = new ListNode(0), ptr2 = dummy2;
		for (int i = 0; i < len2; i++) {
			ptr2.next = new ListNode(r.nextInt(10));
			ptr2 = ptr2.next;
		}

		ListNode l1 = dummy1.next, l2 = dummy2.next;
		printList(l1);
		printList(l2);
		ListNode sum = addList(l1, l2);
		printList(sum);
	}
}
