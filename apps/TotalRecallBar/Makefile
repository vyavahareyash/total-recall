SWIFTC ?= swiftc
TARGET ?= TotalRecallBar
SRC ?= main.swift
CACHE ?= /tmp/swift-cache

.PHONY: all build clean run

all: build

build:
	mkdir -p $(CACHE)
	$(SWIFTC) -module-cache-path $(CACHE) -O $(SRC) -o $(TARGET) -framework Cocoa -framework WebKit
	@ls -lh $(TARGET)

clean:
	rm -f $(TARGET)

run: build
	./$(TARGET) http://127.0.0.1:8765
