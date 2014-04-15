import org.junit.Test;


public class IsPrime {
	public boolean isPrime(int k) {
		if ((k & 1) == 0) {
			return false;
		}
		
		for (int i = 3; i * i <= k; i += 2) {
			if (k % i == 0) {
				return false;
			}
		}
		
		return true;
	}
	
	@Test
	public void test() {
		for (int i = 0; i < 1001; i++) {
			if (isPrime(i)) {
				System.out.println(i);
			}
		}
	}
}
