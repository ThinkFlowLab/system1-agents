import AppKit
import Foundation

private let output = URL(fileURLWithPath: "/tmp/s1a-visual-target.txt")

// The tiles are drawn pixels, deliberately absent from the accessibility tree.
final class TileCanvas: NSView {
    var saveOnLeft = true
    var selected: ((Bool) -> Void)?
    override var isFlipped: Bool { true }
    private let tiles = [NSRect(x: 60, y: 140, width: 200, height: 100),
                         NSRect(x: 340, y: 140, width: 200, height: 100)]

    override func draw(_ dirtyRect: NSRect) {
        NSColor.windowBackgroundColor.setFill()
        bounds.fill()
        let heading: NSString = "Choose the tile labelled Save"
        heading.draw(at: NSPoint(x: 60, y: 55), withAttributes: [.font: NSFont.systemFont(ofSize: 24),
                                                               .foregroundColor: NSColor.labelColor])
        for (index, tile) in tiles.enumerated() {
            NSColor.controlBackgroundColor.setFill()
            NSBezierPath(roundedRect: tile, xRadius: 12, yRadius: 12).fill()
            let label: NSString = (index == 0) == saveOnLeft ? "Save" : "Cancel"
            let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 28),
                                                            .foregroundColor: NSColor.labelColor]
            let size = label.size(withAttributes: attributes)
            label.draw(at: NSPoint(x: tile.midX - size.width / 2, y: tile.midY - size.height / 2),
                       withAttributes: attributes)
        }
    }

    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        if let index = tiles.firstIndex(where: { $0.contains(point) }) {
            selected?((index == 0) == saveOnLeft)
        }
    }
}

final class VisualFixture: NSObject, NSApplicationDelegate {
    private let window = NSWindow(contentRect: NSRect(x: 300, y: 300, width: 600, height: 400),
                                  styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
    private let canvas = TileCanvas(frame: NSRect(x: 0, y: 0, width: 600, height: 400))
    private let status = NSTextView(frame: .zero)

    func applicationDidFinishLaunching(_ notification: Notification) {
        window.title = "S1A Visual Fixture"
        canvas.setAccessibilityElement(false)
        status.isEditable = false
        status.isSelectable = false
        status.drawsBackground = false
        status.string = "Waiting"
        status.setAccessibilityLabel("Status")
        status.frame = NSRect(x: 300, y: 315, width: 230, height: 35)
        let reset = NSButton(title: "Reset", target: self, action: #selector(resetTask))
        reset.frame = NSRect(x: 60, y: 315, width: 120, height: 35)
        canvas.addSubview(reset)
        canvas.addSubview(status)
        canvas.selected = { correct in
            do {
                let previous = (try? String(contentsOf: output, encoding: .utf8)) ?? ""
                let result = correct ? "Save selected\n" : "Cancel selected\n"
                try (previous + result).write(to: output, atomically: true, encoding: .utf8)
                self.status.string = correct ? "Saved" : "Wrong tile"
            } catch {
                self.status.string = "Write failed"
            }
        }
        window.contentView = canvas
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func resetTask() {
        status.string = "Waiting"
        canvas.saveOnLeft.toggle()
        canvas.needsDisplay = true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let delegate = VisualFixture()
app.delegate = delegate
app.run()
