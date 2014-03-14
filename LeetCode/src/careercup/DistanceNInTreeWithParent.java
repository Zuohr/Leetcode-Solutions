package careercup;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class DistanceNInTreeWithParent {
	public ArrayList<TreeNode> solveBFS(TreeNode node, int n) {
		ArrayList<TreeNode> result = new ArrayList<TreeNode>();
		if (node == null || n < 0) {
			return result;
		}
		
		Set<TreeNode> src = new HashSet<TreeNode>(), dst = new HashSet<TreeNode>();
		Set<TreeNode> vst = new HashSet<TreeNode>();
		src.add(node);
		int count = 0;
		while (!src.isEmpty() && count < n) {
			for (TreeNode curr : src) {
				if (vst.contains(curr)) {
					continue;
				}
				vst.add(curr);
				if (curr.left != null && !vst.contains(curr.left)) {
					dst.add(curr.left);
				}
				if (curr.right != null && !vst.contains(curr.right)) {
					dst.add(curr.right);
				}
				if (curr.parent != null && !vst.contains(curr.parent)) {
					dst.add(curr.parent);
				}
			}
			
			count++;
			src.clear();
			Set<TreeNode> tmp = src;
			src = dst;
			dst = tmp;
		}
		
		for (TreeNode curr : src) {
			result.add(curr);
		}
		
		return result;
	}
	
	public ArrayList<TreeNode> solveDFS(TreeNode node, int n) {
		ArrayList<TreeNode> result = new ArrayList<TreeNode>();
		if (node == null || n < 0) {
			return result;
		}
		
		Set<TreeNode> vst = new HashSet<TreeNode>();
		vst.add(node);
		solveDFS(result, vst, node, n);
		
		return result;
	}

	private void solveDFS(ArrayList<TreeNode> result, Set<TreeNode> vst,
			TreeNode node, int n) {
		if (n == 0) {
			result.add(node);
			return;
		}
		
		if (node.left != null && !vst.contains(node.left)) {
			vst.add(node.left);
			solveDFS(result, vst, node.left, n - 1);
			vst.remove(node.left);
		}
		if (node.right != null && !vst.contains(node.right)) {
			vst.add(node.right);
			solveDFS(result, vst, node.right, n - 1);
			vst.remove(node.right);
		}
		if (node.parent != null && !vst.contains(node.parent)) {
			vst.add(node.parent);
			solveDFS(result, vst, node.parent, n - 1);
			vst.remove(node.parent);
		}
	}
}

class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;
	TreeNode parent;
}