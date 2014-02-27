import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Q5 {
	private static HashMap<Long, ArrayList<Node>> map = new HashMap<Long, ArrayList<Node>>();
	
	private static class Node {
		private final int id;
		private final List<Node> children;

		Node(int id) {
			this.id = id;
			this.children = new ArrayList<Node>();
		}

		@Override
		public String toString() {
			return String.valueOf(id);
		}
	}

	private static List<Node> getLargestCommonSubtrees(Node root) {
		List<Node> result = new ArrayList<Node>();
		if (root == null || root.children == null) {
			return result;
		}
		
		
		return result;
	}
	
	private static Long getHash(Node root) {
		if (root == null) {
			return 0l;
		}
		
		
		return 0l;
	}
}
