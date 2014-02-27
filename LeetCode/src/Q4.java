import java.util.ArrayList;
import java.util.List;

public class Q4 {
	public static List<Integer> windowSum(List<Integer> list, int windowSize) {
		// Handling exceptions
		if (list == null || windowSize <= 0 || windowSize > list.size()) {
			throw new IllegalArgumentException(
					"window size cannot be less than or equal zero or greater than the length of array");
		}

		ArrayList<Integer> result = new ArrayList<Integer>();
		if (list.isEmpty()) {
			return result;
		}
		int tempSum = 0;
		// get the sum of first kth number(k == window size)
		for (int j = 0; j < windowSize; j++) {
			tempSum += list.get(j);
		}
		result.add(tempSum);
		// assign left pointer and right pointer
		int left = 1;
		int right = left + windowSize - 1;
		// sliding window
		while (right < list.size()) {
			tempSum = tempSum - list.get(left - 1) + list.get(right);
			result.add(tempSum);
			left++;
			right++;
		}

		return result;
	}

	public static void main(String[] args) {
		int[] input = { 4, 2, 73, 11, -5 };
		ArrayList<Integer> list = new ArrayList<Integer>();
		for (int n : input) {
			list.add(n);
		}
		for (long i : windowSum(list, 6)) {
			System.out.println(i);
		}
	}
}
