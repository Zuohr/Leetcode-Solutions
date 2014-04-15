import org.junit.Test;

public class test2 {
	public boolean isMatch(String s, String p) {
		if (s == null || p == null) {
			return false;
		}

		int lenS = s.length(), lenP = p.length();
		if (s.isEmpty()) {
			if (lenP > 1 && p.charAt(0) != '*' && p.charAt(1) == '*') {
				return isMatch(s, p.substring(2));
			} else {
				return false;
			}
		} else if (p.isEmpty()) {
			return s.isEmpty();
		}

		char lastS = s.charAt(lenS - 1), lastP = p.charAt(lenP - 1);
		if (lastP != '*' && lastP != '.' && lastS != lastP) {
			return false;
		}

		if (p.charAt(0) == '*') {
			return false;
		}

		if (lenP > 1 && p.charAt(1) == '*') {
			if (p.charAt(0) == '.') {
				for (int i = lenS; i >= 0; i--) {
					if (isMatch(s.substring(i), p.substring(2))) {
						return true;
					}
				}
				return false;
			} else {
				int index = 0;
				char curr = p.charAt(0);
				while (index < lenS && s.charAt(index) == curr) {
					index++;
				}
				for (int i = index; i >= 0; i--) {
					if (isMatch(s.substring(i), p.substring(2))) {
						return true;
					}
				}
				return false;
			}
		} else {
			if (p.charAt(0) == '.') {
				return isMatch(s.substring(1), p.substring(1));
			} else {
				return p.charAt(0) == s.charAt(0)
						&& isMatch(s.substring(1), p.substring(1));
			}
		}
	}

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
	
	@Test
	public void test() {
		System.out.println(isMatch("baccbbcbcacacbbc", "c*.*b*c*ba*b*b*.a*"));
	}
}
