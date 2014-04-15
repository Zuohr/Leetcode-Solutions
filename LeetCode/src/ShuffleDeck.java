import java.util.Arrays;
import java.util.Random;

import org.junit.Test;


public class ShuffleDeck {
	public void shuffle(int[] arr) {
		if (arr == null) {
			return;
		}
		
		Random random = new Random();
		for (int i = 0; i < arr.length; i++) {
			int r = random.nextInt(i + 1);
			int temp = arr[i];
			arr[i] = arr[r];
			arr[r] = temp;
		}
	}
	
	@Test
	public void test() {
		int[] arr = { 1, 2, 3, 4, 5 };
		shuffle(arr);
		System.out.println(Arrays.toString(arr));
	}
}
