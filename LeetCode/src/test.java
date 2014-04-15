import org.junit.Test;

public class test {
	public static void main(String[] args) {
		
	}

	public static ListNode customersToNotify(ListNode myList, int k) {
		ListNode dummy = new ListNode();
		dummy.next = myList;

		ListNode back = dummy, front = dummy;
		for (int i = 0; i < k; i++) {
			front = front.next;
		}

		while (front != null) {
			front = front.next;
			back = back.next;
		}

		return back;
	}

	@Test
	public void testist() {
		ListNode head = new ListNode();
		head.customerId = 0;
		ListNode ptr = head;

		for (int i = 0; i < 5; i++) {
			ptr.next = new ListNode();
			ptr = ptr.next;
			ptr.customerId = i;
		}
		ptr = head;
		print(ptr);
		if (ptr.customerId == 0) {
			throw new IllegalArgumentException("");
		}
		System.out.println();
		print(customersToNotify(head, 6));
	}

	public void print(ListNode node) {
		while (node != null) {
			System.out.print(node.customerId + " -> ");
			node = node.next;
		}
	}

}

class ListNode {
	public int customerId;
	public ListNode next;
}
