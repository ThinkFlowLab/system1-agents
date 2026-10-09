import AppKit
import Foundation

// A small local app for the desktop agent's macOS end-to-end test. The file is
// written only when the app's Save button is pressed through the GUI.
private let output = URL(fileURLWithPath: "/tmp/s1a-desktop-fixture.txt")

final class DocumentFixture: NSObject, NSApplicationDelegate {
    private let window = NSWindow(
        contentRect: NSRect(x: 300, y: 300, width: 620, height: 440),
        styleMask: [.titled, .closable, .miniaturizable, .resizable],
        backing: .buffered,
        defer: false
    )
    private let editor = NSTextView(frame: NSRect(x: 0, y: 0, width: 580, height: 320))
    private let status = NSTextView(frame: NSRect(x: 210, y: 25, width: 360, height: 22))

    func applicationDidFinishLaunching(_ notification: Notification) {
        window.title = "S1A Document Fixture"
        let content = NSView(frame: window.contentView!.bounds)
        content.autoresizingMask = [.width, .height]

        let scroll = NSScrollView(frame: NSRect(x: 20, y: 75, width: 580, height: 340))
        scroll.hasVerticalScroller = true
        scroll.autoresizingMask = [.width, .height]
        editor.isEditable = true
        editor.isSelectable = true
        editor.setAccessibilityLabel("Body")
        scroll.documentView = editor
        content.addSubview(scroll)

        let save = NSButton(title: "Save", target: self, action: #selector(saveDocument))
        save.frame = NSRect(x: 20, y: 20, width: 80, height: 30)
        content.addSubview(save)

        let clear = NSButton(title: "Clear", target: self, action: #selector(clearDocument))
        clear.frame = NSRect(x: 110, y: 20, width: 80, height: 30)
        content.addSubview(clear)

        status.frame = NSRect(x: 210, y: 25, width: 360, height: 22)
        status.isEditable = false
        status.isSelectable = false
        status.drawsBackground = false
        status.string = "Editing"
        status.setAccessibilityLabel("Status")
        content.addSubview(status)

        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func saveDocument() {
        do {
            try editor.string.write(to: output, atomically: true, encoding: .utf8)
            status.string = "Saved"
        } catch {
            status.string = "Save failed"
        }
    }

    @objc private func clearDocument() {
        editor.string = ""
        status.string = "Editing"
        try? FileManager.default.removeItem(at: output)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

let app = NSApplication.shared
let delegate = DocumentFixture()
app.delegate = delegate
app.run()
