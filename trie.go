// Implementation of an R-Way Trie data structure.
//
// A Trie has a root Node which is the base of the tree.
// Each subsequent Node has a letter and children, which are
// nodes that have letter values associated with them.
package trie

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/golang/glog"
)

const (
	dirSep = '/'
)

type Node struct {
	val       rune
	path      string
	term      bool
	depth     int
	meta      interface{}
	mask      uint64
	parent    *Node
	children  map[rune]*Node
	termCount int
}

type Trie struct {
	mu   sync.RWMutex
	root *Node
	size int
}

type ByKeys []string

func (a ByKeys) Len() int           { return len(a) }
func (a ByKeys) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByKeys) Less(i, j int) bool { return len(a[i]) < len(a[j]) }

const nul = 0x0

type WalkFunc func(node *Node, name, path string) bool

// Creates a new Trie with an initialized root Node.
func New() *Trie {
	return &Trie{
		root: &Node{children: make(map[rune]*Node), depth: 0},
		size: 0,
	}
}

// Returns the root node for the Trie.
func (t *Trie) Root() *Node {
	return t.root
}

// Adds the key to the Trie, including meta data. Meta data
// is stored as `interface{}` and must be type cast by
// the caller.
func (t *Trie) Add(key string, meta interface{}) *Node {
	return t.AddAtNode(key, t.root, meta)
}
func (t *Trie) AddAtNode(key string, node *Node, meta interface{}) *Node {
	if node.Terminating() {
		return t.addAtNode(key, node.parent, node.path, meta)
	}
	return t.addAtNode(key, node, node.path, meta)
}
func (t *Trie) addAtNode(key string, node *Node, path string, meta interface{}) *Node {
	t.mu.Lock()

	t.size++
	runes := []rune(key)
	bitmask := maskruneslice(runes)
	node.mask |= bitmask
	node.termCount++
	for i := range runes {
		r := runes[i]
		bitmask = maskruneslice(runes[i:])
		if n, ok := node.children[r]; ok {
			node = n
			node.mask |= bitmask
		} else {
			node = node.NewChild(r, "", bitmask, nil, false)
		}
		node.termCount++
	}
	node = node.NewChild(nul, path+key, 0, meta, true)
	t.mu.Unlock()

	return node
}

// Finds and returns meta data associated
// with `key`.
func (t *Trie) Find(key string) (*Node, bool) {
	return t.FindAtNode(key, t.Root())
}

// Finds and returns meta data associated
// with `key` relative to n.
func (t *Trie) FindAtNode(key string, n *Node) (*Node, bool) {
	if n.Terminating() {
		return t.findAtNode(key, n.parent)
	}
	return t.findAtNode(key, n)
}

// Finds and returns meta data associated
// with `key` relative to n.
func (t *Trie) findAtNode(key string, n *Node) (*Node, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	node := findNode(n, []rune(key))
	if node == nil {
		return nil, false
	}

	node, ok := node.Children()[nul]
	if !ok || !node.term {
		return nil, false
	}

	return node, true
}

func (t *Trie) HasKeysWithPrefix(key string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	node := findNode(t.Root(), []rune(key))
	return node != nil
}

// Removes a key from the trie relative to node. Node must be terminating
// Hence we use 'parent' so it can search for key relative to it.
func (t *Trie) RemoveAtNode(key string, node *Node) {
	if node.Terminating() {
		t.removeAtNode(key, node)
	}

	t.removeAtNode(key, node.parent)
}

// Removes a key from the trie, ensuring that
// all bitmasks up to root are appropriately recalculated.
func (t *Trie) Remove(key string) {
	t.removeAtNode(key, t.Root())
}

// Removes a key from the trie relative to node, ensuring that
// all bitmasks up to root are appropriately recalculated.
func (t *Trie) removeAtNode(key string, start *Node) {
	if key == "" {
		return
	}
	var (
		i    int
		rs   = []rune(key)
		node = findNode(start, []rune(key))
	)
	if node == nil {
		return
	}
	t.mu.Lock()

	t.size--
	for n := node.Parent(); n != nil; n = n.Parent() {
		i++
		if len(n.Children()) > 1 {
			r := rs[len(rs)-i]
			n.RemoveChild(r)
			break
		}
	}
	t.mu.Unlock()
}

// Returns all the keys currently stored in the trie.
func (t *Trie) Keys() []string {
	if t.size == 0 {
		return []string{}
	}

	return t.PrefixSearch("")
}

// Performs a fuzzy search against the keys in the trie.
func (t Trie) FuzzySearch(pre string) []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	keys := fuzzycollect(t.Root(), []rune(pre))
	sort.Sort(ByKeys(keys))
	return keys
}

// Performs a prefix search against the keys in the trie.
func (t Trie) PrefixSearch(pre string) []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	node := findNode(t.Root(), []rune(pre))
	if node == nil {
		return nil
	}

	return collect(node)
}

func (t Trie) ExactSearchAtNode(s string, n *Node) ([]string, []*Node, error) {
	res := make([]string, 0)
	nodes := make([]*Node, 0)
	walker := func(node *Node, name, path string) bool {
		if strings.EqualFold(s, name) {
			res = append(res, path)
			nodes = append(nodes, node)
		}
		return true
	}
	if err := t.WalkAtNode(n, walker, true); err != nil {
		return nil, nil, err
	}
	return res, nodes, nil
}

func (t Trie) FirstRegexMatchAtNode(s string, n *Node) (string, *Node, error) {
	res := ""
	var found *Node = nil
	re, err := regexp.Compile(s)
	if err != nil {
		return "", nil, err
	}
	walker := func(node *Node, name, path string) bool {
		if re.MatchString(name) {
			found = node
			res = path
			return false
		}
		return true
	}
	if err := t.WalkAtNode(n, walker, true); err != nil {
		return "", nil, err
	}
	return res, found, nil
}

func (t Trie) ListAtNode(n *Node) ([]string, []*Node, error) {
	// Walk children until we find a dirSep, but don't recurse.
	res := make([]string, 0)
	nodes := make([]*Node, 0)
	walker := func(node *Node, name, path string) bool {
		res = append(res, name)
		nodes = append(nodes, node)
		return true
	}
	if err := t.WalkAtNode(n, walker, false); err != nil {
		return nil, nil, err
	}
	return res, nodes, nil
}

func (t Trie) WalkAtNode(n *Node, walker WalkFunc, nested bool) error {
	if !n.Terminating() || n.Parent().Val() != dirSep {
		return fmt.Errorf("Node must be terminating and a dir")
	}

	t.mu.RLock()
	defer t.mu.RUnlock()
	name := make([]rune, 0)
	t.walkAtNode(n.parent, walker, name, nested)
	return nil
}

func (t Trie) walkAtNode(n *Node, walker WalkFunc, name []rune, nested bool) bool {
	glog.V(2).Infof("Walking node: %+v, name: %s\n", n, string(name))
	if n.parent.val == dirSep {
		// Reset name after every nested directory.
		name = []rune{n.val}
	}
	for r, child := range n.children {
		if child.Terminating() {
			if len(name) > 0 { // Exclude root, which is empty
				if !walker(child, string(name), child.path) {
					return false
				}
			}
		} else {
			if r == dirSep {
				if nested {
					if !t.walkAtNode(child, walker, name, nested) {
						return false
					}
				} else {
					if !walker(child.children[nul], string(name), child.children[nul].path) {
						return false
					}
				}
			} else {
				name = append(name, r)
				if !t.walkAtNode(child, walker, name, nested) {
					return false
				}
				name = name[:len(name)-1]
			}
		}
	}
	return true
}

// Creates and returns a pointer to a new child for the node.
func (parent *Node) NewChild(val rune, path string, bitmask uint64, meta interface{}, term bool) *Node {
	node := &Node{
		val:      val,
		path:     path,
		mask:     bitmask,
		term:     term,
		meta:     meta,
		parent:   parent,
		children: make(map[rune]*Node),
		depth:    parent.depth + 1,
	}
	parent.children[node.val] = node
	parent.mask |= bitmask
	return node
}

func (n *Node) RemoveChild(r rune) {
	delete(n.children, r)
	for nd := n.parent; nd != nil; nd = nd.parent {
		nd.mask ^= nd.mask
		nd.mask |= uint64(1) << uint64(nd.val-'a')
		for _, c := range nd.children {
			nd.mask |= c.mask
		}
	}
}

// Returns the parent of this node.
func (n Node) Parent() *Node {
	return n.parent
}

// Returns the meta information of this node.
func (n Node) Meta() interface{} {
	return n.meta
}

// Returns the children of this node.
func (n Node) Children() map[rune]*Node {
	return n.children
}

func (n Node) Terminating() bool {
	return n.term
}

func (n Node) Val() rune {
	return n.val
}

func (n Node) Depth() int {
	return n.depth
}

func (n Node) Path() string {
	return n.path
}

func (n Node) Name() string {
	path := []rune(n.path)
	if len(path) > 0 && path[len(path)-1] == dirSep {
		path = path[:len(path)-1]
	}
	idx := strings.LastIndex(string(path), "/")
	return string(path[idx+1:])
}

// Returns a uint64 representing the current
// mask of this node.
func (n Node) Mask() uint64 {
	return n.mask
}

func findNode(node *Node, runes []rune) *Node {
	glog.V(2).Infof("find node %+v - %s.\n", node, string(runes))
	if node == nil {
		return nil
	}

	if len(runes) == 0 {
		return node
	}

	n, ok := node.Children()[runes[0]]
	if !ok {
		return nil
	}

	var nrunes []rune
	if len(runes) > 1 {
		nrunes = runes[1:]
	} else {
		nrunes = runes[0:0]
	}

	return findNode(n, nrunes)
}

func maskruneslice(rs []rune) uint64 {
	var m uint64
	for _, r := range rs {
		m |= uint64(1) << uint64(r-'a')
	}
	return m
}

func collect(node *Node) []string {
	var (
		n *Node
		i int
	)
	keys := make([]string, 0, node.termCount)
	nodes := make([]*Node, 1, len(node.children))
	nodes[0] = node
	for l := len(nodes); l != 0; l = len(nodes) {
		i = l - 1
		n = nodes[i]
		nodes = nodes[:i]
		for _, c := range n.children {
			nodes = append(nodes, c)
		}
		if n.term {
			word := n.path
			keys = append(keys, word)
		}
	}
	return keys
}

type potentialSubtree struct {
	idx  int
	node *Node
}

func fuzzycollect(node *Node, partial []rune) []string {
	if len(partial) == 0 {
		return collect(node)
	}

	var (
		m    uint64
		i    int
		p    potentialSubtree
		keys []string
	)

	potential := []potentialSubtree{potentialSubtree{node: node, idx: 0}}
	for l := len(potential); l > 0; l = len(potential) {
		i = l - 1
		p = potential[i]
		potential = potential[:i]
		m = maskruneslice(partial[p.idx:])
		if (p.node.mask & m) != m {
			continue
		}

		if p.node.val == partial[p.idx] {
			p.idx++
			if p.idx == len(partial) {
				keys = append(keys, collect(p.node)...)
				continue
			}
		}

		for _, c := range p.node.children {
			potential = append(potential, potentialSubtree{node: c, idx: p.idx})
		}
	}
	return keys
}
