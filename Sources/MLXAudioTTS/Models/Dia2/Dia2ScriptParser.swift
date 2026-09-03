import Foundation

/// Minimal tokenizer surface, so the parser is testable without loading
/// swift-transformers and its 49k-entry vocabulary.
public protocol Dia2TextTokenizing: Sendable {
    func encode(_ text: String) -> [Int]
}

/// Splits a tagged script into per-word scheduling entries.
/// Ported from dia2/runtime/script_parser.py.
public struct Dia2ScriptParser: Sendable {
    public let tokenIDs: Dia2TokenIDs
    public let frameRate: Double
    /// Frames of slack added after each word, before its own token count.
    private let paddingBetween = 1

    public init(tokenIDs: Dia2TokenIDs, frameRate: Double) {
        self.tokenIDs = tokenIDs
        self.frameRate = frameRate
    }

    public func parse(_ lines: [String], tokenizer: any Dia2TextTokenizing) -> [Dia2Entry] {
        var entries: [Dia2Entry] = []
        var lastSpeaker: Int?

        for (index, rawLine) in lines.enumerated() {
            let normalized = rawLine
                .replacingOccurrences(of: "\u{2019}", with: "'")
                .replacingOccurrences(of: ":", with: " ")
            var pendingSpeaker: Int?
            var firstContent = true

            for chunk in splitOnBreaks(normalized) {
                switch chunk {
                case .pause(let seconds):
                    let padding = Int((seconds * frameRate).rounded())
                    if padding > 0 {
                        entries.append(Dia2Entry(tokens: [], text: "", padding: padding))
                    }
                case .text(let segment):
                    for word in segment.split(separator: " ").map(String.init) {
                        if word == "[S1]" { pendingSpeaker = tokenIDs.spk1; continue }
                        if word == "[S2]" { pendingSpeaker = tokenIDs.spk2; continue }
                        var tokens = pendingSpeaker != nil
                            ? tokenizer.encode("\(pendingSpeaker == tokenIDs.spk1 ? "[S1]" : "[S2]") \(word)")
                            : tokenizer.encode(word)
                        if firstContent {
                            // Alternate by line when the script omits tags, and
                            // never repeat a tag the tokenizer already produced.
                            let speaker = pendingSpeaker ?? (index % 2 == 0 ? tokenIDs.spk1 : tokenIDs.spk2)
                            if lastSpeaker != speaker, tokens.first != speaker {
                                tokens.insert(speaker, at: 0)
                            }
                            lastSpeaker = speaker
                            firstContent = false
                        }
                        pendingSpeaker = nil
                        let padding = max(0, paddingBetween + tokens.count - 1)
                        entries.append(Dia2Entry(tokens: tokens, text: word, padding: padding))
                    }
                }
            }
        }
        return entries
    }

    private enum Chunk { case text(String), pause(Double) }

    private func splitOnBreaks(_ line: String) -> [Chunk] {
        let pattern = #"<break\s+time="([0-9]+(?:\.[0-9]*)?)s"\s*/?>"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [.text(line)] }
        let ns = line as NSString
        var chunks: [Chunk] = []
        var cursor = 0
        for match in regex.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > cursor {
                chunks.append(.text(ns.substring(with: NSRange(location: cursor,
                                                              length: match.range.location - cursor))))
            }
            if let seconds = Double(ns.substring(with: match.range(at: 1))) {
                chunks.append(.pause(seconds))
            }
            cursor = match.range.location + match.range.length
        }
        if cursor < ns.length {
            chunks.append(.text(ns.substring(from: cursor)))
        }
        return chunks
    }
}
