//! `align_words` of the Python server: where in a library text a quotation is, and how its
//! words differ. Built on a port of Python's `difflib.SequenceMatcher` (no junk,
//! `autojunk=False`), so the matching blocks and opcodes are the ones Python computes.

use std::collections::{HashMap, HashSet};

use super::textutil::norm_word;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub a: usize,
    pub b: usize,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Equal,
    Replace,
    Delete,
    Insert,
}

/// `difflib.SequenceMatcher(None, a, b, autojunk=False)`.
pub struct SequenceMatcher<'s> {
    a: &'s [String],
    b: &'s [String],
    b2j: HashMap<&'s str, Vec<usize>>,
}

impl<'s> SequenceMatcher<'s> {
    pub fn new(a: &'s [String], b: &'s [String]) -> Self {
        let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
        for (j, x) in b.iter().enumerate() {
            b2j.entry(x.as_str()).or_default().push(j);
        }
        SequenceMatcher { a, b, b2j }
    }

    fn find_longest_match(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> Block {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = self.b2j.get(self.a[i].as_str()) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j
                        .checked_sub(1)
                        .and_then(|p| j2len.get(&p))
                        .copied()
                        .unwrap_or(0)
                        + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        // With no junk the extensions difflib makes add nothing beyond the loop above,
        // except across equal runs it split; kept as written for fidelity.
        while besti > alo && bestj > blo && self.a[besti - 1] == self.b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && self.a[besti + bestsize] == self.b[bestj + bestsize]
        {
            bestsize += 1;
        }
        Block {
            a: besti,
            b: bestj,
            size: bestsize,
        }
    }

    pub fn get_matching_blocks(&self) -> Vec<Block> {
        let (la, lb) = (self.a.len(), self.b.len());
        let mut queue = vec![(0, la, 0, lb)];
        let mut blocks = Vec::new();
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let m = self.find_longest_match(alo, ahi, blo, bhi);
            if m.size > 0 {
                blocks.push(m);
                if alo < m.a && blo < m.b {
                    queue.push((alo, m.a, blo, m.b));
                }
                if m.a + m.size < ahi && m.b + m.size < bhi {
                    queue.push((m.a + m.size, ahi, m.b + m.size, bhi));
                }
            }
        }
        blocks.sort_by_key(|x| (x.a, x.b, x.size));
        // collapse adjacent blocks
        let mut out: Vec<Block> = Vec::new();
        let (mut i1, mut j1, mut k1) = (0, 0, 0);
        for blk in blocks {
            if i1 + k1 == blk.a && j1 + k1 == blk.b {
                k1 += blk.size;
            } else {
                if k1 > 0 {
                    out.push(Block {
                        a: i1,
                        b: j1,
                        size: k1,
                    });
                }
                i1 = blk.a;
                j1 = blk.b;
                k1 = blk.size;
            }
        }
        if k1 > 0 {
            out.push(Block {
                a: i1,
                b: j1,
                size: k1,
            });
        }
        out.push(Block {
            a: la,
            b: lb,
            size: 0,
        });
        out
    }

    pub fn get_opcodes(&self) -> Vec<(Tag, usize, usize, usize, usize)> {
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::new();
        for blk in self.get_matching_blocks() {
            let tag = match (i < blk.a, j < blk.b) {
                (true, true) => Some(Tag::Replace),
                (true, false) => Some(Tag::Delete),
                (false, true) => Some(Tag::Insert),
                (false, false) => None,
            };
            if let Some(t) = tag {
                out.push((t, i, blk.a, j, blk.b));
            }
            i = blk.a + blk.size;
            j = blk.b + blk.size;
            if blk.size > 0 {
                out.push((Tag::Equal, blk.a, i, blk.b, j));
            }
        }
        out
    }
}

const PREFIXES: [char; 8] = ['ו', 'ב', 'ד', 'ה', 'כ', 'ל', 'מ', 'ש'];

fn has_quote_mark(t: &str) -> bool {
    t.contains(['"', '\u{05F4}', '\'', '\u{05F3}'])
}

/// `_initials_variants`: the letter strings a span may abbreviate to. An abbreviation token
/// (with gershayim) gives its letters, with or without a one-letter prefix; a plain word its
/// first letter, or its first two when the first may be a prefix (דרב הונא ~ דר"ה).
fn initials_variants(tokens: &[String]) -> HashSet<String> {
    let mut opts: Vec<Vec<String>> = Vec::new();
    for t in tokens {
        let n = norm_word(t);
        let c: Vec<char> = n.chars().collect();
        if c.is_empty() {
            continue;
        }
        let mut v: Vec<String> = Vec::new();
        if has_quote_mark(t) {
            v.push(n.clone());
            if c.len() > 2 && PREFIXES.contains(&c[0]) {
                v.push(c[1..].iter().collect());
            }
        } else {
            v.push(c[0].to_string());
            if c.len() > 1 && PREFIXES.contains(&c[0]) {
                v.push(c[..2].iter().collect());
            }
        }
        v.sort();
        v.dedup();
        opts.push(v);
    }
    if opts.is_empty() || opts.len() > 8 {
        return HashSet::new();
    }
    let mut acc: Vec<String> = vec![String::new()];
    for o in &opts {
        let mut next = Vec::with_capacity(acc.len() * o.len());
        for prefix in &acc {
            for piece in o {
                next.push(format!("{prefix}{piece}"));
            }
        }
        acc = next;
    }
    acc.into_iter().collect()
}

/// `_abbr_equiv`: a span of the quote and one of the library text are the same words, one
/// side abbreviated.
fn abbr_equiv(q: &[String], c: &[String]) -> bool {
    if q.is_empty() || c.is_empty() || (q.len() == 1 && c.len() == 1) {
        return false;
    }
    let has_abbr = q.iter().chain(c).any(|t| t.contains(['"', '\u{05F4}']));
    if !has_abbr {
        return false;
    }
    let a = initials_variants(q);
    !a.is_empty() && initials_variants(c).iter().any(|x| a.contains(x))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Replace,
    Delete,
    Insert,
    Abbr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alignment {
    pub ratio: f64,
    pub start: usize,
    pub end: usize,
    /// (op, i1, i2, j1, j2): quote words `i1..i2` against window words `j1..j2`.
    pub ops: Vec<(Op, usize, usize, usize, usize)>,
}

/// `align_words`: the best window of `cw` for the quote `qw` (normalized words; `qo`/`co` the
/// original tokens, used to resolve abbreviations).
pub fn align_words(
    qw: &[String],
    cw: &[String],
    qo: &[String],
    co: &[String],
) -> Option<Alignment> {
    if qw.is_empty() || cw.is_empty() {
        return None;
    }
    let sm = SequenceMatcher::new(qw, cw);
    let blocks: Vec<Block> = sm
        .get_matching_blocks()
        .into_iter()
        .filter(|b| b.size > 0)
        .collect();
    if blocks.is_empty() {
        return None;
    }
    // keep the densest run of matches: a stray common word far away must not stretch the window
    let mut clusters: Vec<Vec<Block>> = Vec::new();
    let mut cur = vec![blocks[0]];
    for &blk in &blocks[1..] {
        let prev = *cur.last().unwrap();
        let gap_b = blk.b as i64 - (prev.b + prev.size) as i64;
        let gap_a = blk.a as i64 - (prev.a + prev.size) as i64;
        if gap_b <= gap_a + 4 {
            cur.push(blk);
        } else {
            clusters.push(std::mem::take(&mut cur));
            cur = vec![blk];
        }
    }
    clusters.push(cur);
    // max() keeps the first of equal clusters
    let mut best = &clusters[0];
    let weight = |c: &Vec<Block>| c.iter().map(|x| x.size).sum::<usize>();
    for c in &clusters[1..] {
        if weight(c) > weight(best) {
            best = c;
        }
    }
    let (a_first, b_first) = (best[0].a, best[0].b);
    let last = *best.last().unwrap();
    let (a_last, b_last) = (last.a + last.size, last.b + last.size);
    let mut start = b_first.saturating_sub(a_first);
    let mut end = cw.len().min(b_last + (qw.len() - a_last));
    // an abbreviated library text is shorter than the quote: pull the edges in when the
    // unmatched edge of the quote abbreviates to a suffix/prefix of the window's edge
    if a_first > 0 {
        for j in start..b_first {
            if abbr_equiv(&qo[..a_first], &co[j..b_first]) {
                start = j;
                break;
            }
        }
    }
    if a_last < qw.len() {
        let mut j = end;
        while j > b_last {
            if abbr_equiv(&qo[a_last..], &co[b_last..j]) {
                end = j;
                break;
            }
            j -= 1;
        }
    }
    let window = &cw[start..end];
    let w = SequenceMatcher::new(qw, window);
    let (mut ops, mut mq, mut mc) = (Vec::new(), 0usize, 0usize);
    for (tag, i1, i2, j1, j2) in w.get_opcodes() {
        match tag {
            Tag::Equal => {
                mq += i2 - i1;
                mc += j2 - j1;
            }
            Tag::Replace if abbr_equiv(&qo[i1..i2], &co[start + j1..start + j2]) => {
                mq += i2 - i1;
                mc += j2 - j1;
                ops.push((Op::Abbr, i1, i2, j1, j2));
            }
            Tag::Replace => ops.push((Op::Replace, i1, i2, j1, j2)),
            Tag::Delete => ops.push((Op::Delete, i1, i2, j1, j2)),
            Tag::Insert => ops.push((Op::Insert, i1, i2, j1, j2)),
        }
    }
    let ratio = (mq + mc) as f64 / (qw.len() + (end - start)).max(1) as f64;
    Some(Alignment {
        ratio,
        start,
        end,
        ops,
    })
}
