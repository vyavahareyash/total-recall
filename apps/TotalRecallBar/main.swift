import Cocoa
import WebKit

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var popover: NSPopover!
    var eventMonitor: Any?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Run strictly as a background menu bar accessory (no Dock icon)
        NSApp.setActivationPolicy(.accessory)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            if let img = NSImage(systemSymbolName: "bubble.left.and.bubble.right.fill", accessibilityDescription: "WhatsApp Total Recall") {
                img.isTemplate = true
                button.image = img
            } else {
                button.title = "WTR"
            }
            button.toolTip = "WhatsApp Total Recall: Local Conversational Intelligence"
            button.action = #selector(statusItemClicked(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
            button.target = self
        }

        // Determine target server URL
        var targetURLString = "http://127.0.0.1:8765"
        for arg in CommandLine.arguments.dropFirst() {
            if arg.hasPrefix("http://") || arg.hasPrefix("https://") {
                targetURLString = arg
                break
            }
        }

        // Configure WKWebView
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 460, height: 640), configuration: config)
        if let url = URL(string: targetURLString) {
            webView.load(URLRequest(url: url))
        }

        let viewController = NSViewController()
        viewController.view = webView

        popover = NSPopover()
        popover.contentSize = NSSize(width: 460, height: 640)
        popover.behavior = .transient
        popover.contentViewController = viewController

        // Monitor global mouse clicks to close popover when user interacts with other apps
        eventMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] event in
            guard let self = self else { return }
            if self.popover.isShown {
                self.popover.performClose(event)
            }
        }
    }

    @objc func statusItemClicked(_ sender: AnyObject?) {
        guard let event = NSApp.currentEvent else {
            togglePopover(sender)
            return
        }

        // Handle right-click with quick contextual menu
        if event.type == .rightMouseUp {
            let menu = NSMenu()
            menu.addItem(NSMenuItem(title: "Open Total Recall Window", action: #selector(openPopover), keyEquivalent: "o"))
            menu.addItem(NSMenuItem.separator())
            menu.addItem(NSMenuItem(title: "Quit Total Recall", action: #selector(quitApp), keyEquivalent: "q"))
            statusItem.menu = menu
            statusItem.button?.performClick(nil)
            statusItem.menu = nil // Reset so regular click resumes popover
        } else {
            togglePopover(sender)
        }
    }

    @objc func openPopover() {
        guard let button = statusItem.button else { return }
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func togglePopover(_ sender: AnyObject?) {
        guard let button = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(sender)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    @objc func quitApp() {
        NSApplication.shared.terminate(self)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
