SWIFTC ?= swiftc
TARGET ?= TotalRecallBar
SRC ?= main.swift
CACHE ?= /tmp/swift-cache
APP_NAME ?= TotalRecallBar.app
DMG_NAME ?= TotalRecallBar-macos.dmg
DMG_STAGING ?= /tmp/tr-dmg-staging

.PHONY: all build universal app dmg clean run

all: build

build:
	mkdir -p $(CACHE)
	$(SWIFTC) -module-cache-path $(CACHE) -O $(SRC) -o $(TARGET) -framework Cocoa -framework WebKit
	@ls -lh $(TARGET)

universal:
	mkdir -p $(CACHE)
	$(SWIFTC) -module-cache-path $(CACHE) -target arm64-apple-macos12.0 -O $(SRC) -o $(TARGET)-arm64 -framework Cocoa -framework WebKit
	$(SWIFTC) -module-cache-path $(CACHE) -target x86_64-apple-macos12.0 -O $(SRC) -o $(TARGET)-x86_64 -framework Cocoa -framework WebKit
	lipo -create -output $(TARGET) $(TARGET)-arm64 $(TARGET)-x86_64
	rm -f $(TARGET)-arm64 $(TARGET)-x86_64
	@file $(TARGET)
	@ls -lh $(TARGET)

app: build
	rm -rf $(APP_NAME)
	mkdir -p $(APP_NAME)/Contents/MacOS
	mkdir -p $(APP_NAME)/Contents/Resources
	cp $(TARGET) $(APP_NAME)/Contents/MacOS/$(TARGET)
	cp Info.plist $(APP_NAME)/Contents/Info.plist
	@echo "Created $(APP_NAME)"

dmg: universal app
	rm -rf $(DMG_STAGING) $(DMG_NAME)
	mkdir -p $(DMG_STAGING)
	cp -R $(APP_NAME) $(DMG_STAGING)/
	ln -s /Applications $(DMG_STAGING)/Applications
	hdiutil create -srcfolder $(DMG_STAGING) -volname "Total Recall" -ov -format UDZO $(DMG_NAME)
	rm -rf $(DMG_STAGING)
	@echo "Created $(DMG_NAME)"
	@shasum -a 256 $(DMG_NAME)

clean:
	rm -rf $(TARGET) $(TARGET)-arm64 $(TARGET)-x86_64 $(APP_NAME) $(DMG_NAME) $(DMG_STAGING)

run: build
	./$(TARGET) http://127.0.0.1:8765
