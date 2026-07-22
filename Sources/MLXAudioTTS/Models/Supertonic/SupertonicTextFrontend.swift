// SuperTonic 3 text frontend — faithful Swift port of the spike's
// `frontend.py`, itself a port of supertone-inc/supertonic `py/helper.py`
// `UnicodeProcessor._preprocess_text`.
//
// Pipeline (must match the reference exactly, or articulation degrades):
// 1. NFKD compatibility decomposition (`decomposedStringWithCompatibilityMapping`,
//    NOT the canonical variant) — Hangul -> jamo, e-acute -> e + combining.
// 2. Emoji removal, dash/quote/symbol replacements, expression expansion.
// 3. Spacing fixes around punctuation, duplicate-quote collapse, whitespace squeeze.
// 4. Append "." if the text doesn't already end in punctuation.
// 5. **Wrap in language tags: `<lang>text</lang>`** — the model is trained with
//    these per-utterance framing tokens; omitting them causes a slurred /
//    "speech impediment" defect even though the words remain intelligible.
// 6. Per-codepoint lookup in the 65,536-entry indexer. Unmapped (-1) codepoints
//    map to embedding row 8321 (the pad/unk row).

import Foundation

final class SupertonicTextFrontend {
    static let unk = 8321

    static let availableLangs: Set<String> = [
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
        "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
        "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
        "vi", "na",
    ]

    private let indexer: [Int]  // 65,536-entry BMP codepoint -> index; -1 = unmapped

    private static let emojiPattern =
        "[\\x{1F600}-\\x{1F64F}\\x{1F300}-\\x{1F5FF}\\x{1F680}-\\x{1F6FF}"
        + "\\x{1F700}-\\x{1F77F}\\x{1F780}-\\x{1F7FF}\\x{1F800}-\\x{1F8FF}"
        + "\\x{1F900}-\\x{1F9FF}\\x{1FA00}-\\x{1FA6F}\\x{1FA70}-\\x{1FAFF}"
        + "\\x{2600}-\\x{26FF}\\x{2700}-\\x{27BF}\\x{1F1E6}-\\x{1F1FF}]+"

    // Ordered: dictionary iteration order is unspecified in Python too, but the
    // replacements are independent so order does not matter; keep source order.
    private static let replacements: [(String, String)] = [
        ("\u{2013}", "-"), ("\u{2011}", "-"), ("\u{2014}", "-"), ("_", " "),
        ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{2018}", "'"), ("\u{2019}", "'"),
        ("\u{00B4}", "'"), ("`", "'"), ("[", " "), ("]", " "), ("|", " "), ("/", " "),
        ("#", " "), ("\u{2192}", " "), ("\u{2190}", " "),
    ]

    private static let expressions: [(String, String)] = [
        ("@", " at "), ("e.g.,", "for example, "), ("i.e.,", "that is, "),
    ]

    init(indexerURL: URL) throws {
        let data = try Data(contentsOf: indexerURL)
        guard let arr = try JSONSerialization.jsonObject(with: data) as? [Int] else {
            throw SupertonicError.invalidInput("unicode_indexer.json is not an int array")
        }
        indexer = arr
    }

    /// Reference `_preprocess_text`: normalize/clean then wrap in <lang> tags.
    func preprocess(_ input: String, lang: String = "en") throws -> String {
        guard Self.availableLangs.contains(lang) else {
            throw SupertonicError.invalidInput("Invalid language: \(lang)")
        }
        var text = input.decomposedStringWithCompatibilityMapping  // NFKD
        text = Self.regexReplace(text, pattern: Self.emojiPattern, with: "")
        for (k, v) in Self.replacements {
            text = text.replacingOccurrences(of: k, with: v)
        }
        text = Self.regexReplace(text, pattern: "[\u{2665}\u{2606}\u{2661}\u{00A9}\\\\]", with: "")
        for (k, v) in Self.expressions {
            text = text.replacingOccurrences(of: k, with: v)
        }
        for p in [" ,", " .", " !", " ?", " ;", " :", " '"] {
            text = text.replacingOccurrences(of: p, with: String(p.dropFirst()))
        }
        while text.contains("\"\"") { text = text.replacingOccurrences(of: "\"\"", with: "\"") }
        while text.contains("''") { text = text.replacingOccurrences(of: "''", with: "'") }
        while text.contains("``") { text = text.replacingOccurrences(of: "``", with: "`") }
        text = Self.regexReplace(text, pattern: "\\s+", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let terminal = "[.!?;:,'\"')\\]}\u{2026}\u{3002}\u{300D}\u{300F}\u{3011}\u{3009}\u{300B}\u{203A}\u{00BB}]$"
        if Self.firstMatchRange(text, pattern: terminal) == nil {
            text += "."
        }
        return "<\(lang)>" + text + "</\(lang)>"
    }

    /// text -> embedding ids (full reference preprocess).
    func encode(_ text: String, lang: String = "en") throws -> [Int] {
        let processed = try preprocess(text, lang: lang)
        // NOTE: iterate unicode scalars, not Characters — combining marks must
        // be looked up as separate codepoints, exactly like Python's `for ch in str`.
        return processed.unicodeScalars.map { scalar in
            let cp = Int(scalar.value)
            let idx = cp < indexer.count ? indexer[cp] : -1
            return idx < 0 ? Self.unk : idx
        }
    }

    // MARK: - Regex helpers (NSRegularExpression handles \x{...} supplementary ranges)

    private static func regexReplace(_ text: String, pattern: String, with replacement: String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return re.stringByReplacingMatches(in: text, range: range, withTemplate: replacement)
    }

    private static func firstMatchRange(_ text: String, pattern: String) -> NSRange? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        return re.firstMatch(in: text, range: range)?.range
    }
}
