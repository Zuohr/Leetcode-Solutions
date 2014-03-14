package careercup;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

public class DistanceNInTreeWOParent {
	public ArrayList<TreeNode> solve(TreeNode root, TreeNode node, int n) {
		ArrayList<TreeNode> result = new ArrayList<TreeNode>();
		if (root == null || n < 0) {
			return result;
		}

		Stack<TreeNode> path = new Stack<TreeNode>();
		boolean[] found = new boolean[1];
		ArrayList<TreeNode> tmp = new ArrayList<TreeNode>();
		tmp.add(root);
		findPath(path, tmp, node, found);
		while (!path.isEmpty() && n >= 0) {
			TreeNode curr = path.pop();
			result.addAll(solve(curr, n--));
		}

		return result;
	}

	private void findPath(Stack<TreeNode> path, ArrayList<TreeNode> tmp,
			TreeNode node, boolean[] found) {
		if (found[0]) {
			return;
		}
		if (tmp.get(tmp.size() - 1) == node) {
			for (TreeNode treeNode : tmp) {
				path.push(treeNode);
				found[0] = true;
				return;
			}
		}

		if (!found[0] && node.left != null) {
			tmp.add(node.left);
			findPath(path, tmp, node.left, found);
			tmp.remove(tmp.size() - 1);
		}
		if (!found[0] && node.right != null) {
			tmp.add(node.right);
			findPath(path, tmp, node.right, found);
			tmp.remove(tmp.size() - 1);
		}
	}

	private ArrayList<TreeNode> solve(TreeNode node, int i) {
		ArrayList<TreeNode> result = new ArrayList<TreeNode>();
		Set<TreeNode> src = new HashSet<TreeNode>(), dst = new HashSet<TreeNode>();
		src.add(node);
		while (!src.isEmpty() && i > 0) {
			for (TreeNode curr : src) {
				if (curr.left != null) {
					dst.add(curr.left);
				}
				if (curr.right != null) {
					dst.add(curr.right);
				}
			}
			
			i--;
			src.clear();
			Set<TreeNode> tmp = src;
			src = dst;
			dst = tmp;
		}
		
		for (TreeNode treeNode : src) {
			result.add(treeNode);
		}
		
		return result;
	}

	class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;
	}
}
