package careercup;

import java.util.ArrayList;
import java.util.PriorityQueue;

import org.junit.Test;

public class KLargest {
	public ArrayList<Integer> getKLargest(ArrayList<Integer> arr, int k) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (arr == null || k <= 0) {
			return result;
		}

		PriorityQueue<Integer> pq = new PriorityQueue<Integer>();
		int i;
		for (i = 0; i < k && i < arr.size(); i++) {
			pq.add(arr.get(i));
		}

		for (; i < arr.size(); i++) {
			int num = arr.get(i);
			if (num > pq.peek()) {
				pq.poll();
				pq.offer(num);
			}
		}
		
		while (!pq.isEmpty()) {
			result.add(pq.poll());
		}

		return result;
	}

	@Test
	public void testKLargest() {
		ArrayList<Integer> arr = new ArrayList<Integer>();
		for (int i = 0; i < 10; i++) {
			arr.add(i);
		}
		ArrayList<Integer> res = getKLargest(arr, 5);
		System.out.println(res);
	}
}
