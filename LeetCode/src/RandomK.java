import java.util.Arrays;
import java.util.Random;

import org.junit.Test;

public class RandomK {
	public int[] getRandomK(int[] arr, int k) {
		if (arr == null || k <= 0) {
			throw new IllegalArgumentException("");
		}

		int[] result = new int[Math.min(arr.length, k)];
		int i;
		for (i = 0; i < result.length; i++) {
			result[i] = arr[i];
		}

		Random random = new Random();
		int[] count = new int[result.length];
		for (; i < arr.length; i++) {
			int r = random.nextInt(arr.length + 1);
			if (r < k) {
				count[r]++;
				result[r] = arr[i];
			}
		}
		System.out.println(Arrays.toString(count));

		return result;
	}

	@Test
	public void test() {
		int[] arr = new int[10000];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = i + 1;
		}
		System.out.println(Arrays.toString(getRandomK(arr, 10)));
	}
}
