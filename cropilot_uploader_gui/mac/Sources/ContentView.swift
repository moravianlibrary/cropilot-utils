import AppKit
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = CropilotViewModel()

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            mainContent
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .alert("Cropilot Uploader", isPresented: $viewModel.showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(viewModel.alertMessage)
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.accentColor)
                Image(systemName: "crop")
                    .font(.system(size: 27, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .frame(width: 54, height: 54)

            VStack(alignment: .leading, spacing: 4) {
                Text("Cropilot Uploader")
                    .font(.system(size: 28, weight: .semibold))
                Text("Nahrajte složku se skeny, zkontrolujte ořezy v Cropilotu a stáhněte hotové originály.")
                    .foregroundStyle(.secondary)
            }

            Spacer()

            StatusBadge(state: viewModel.state)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 22)
    }

    private var mainContent: some View {
        HStack(spacing: 0) {
            formColumn
                .frame(width: 430)
                .padding(24)

            Divider()

            logColumn
                .padding(24)
        }
    }

    private var formColumn: some View {
        VStack(alignment: .leading, spacing: 22) {
            connectionSection

            Picker("", selection: $viewModel.selectedMode) {
                Text("Nahrát").tag(AppMode.upload)
                Text("Stáhnout").tag(AppMode.download)
            }
            .pickerStyle(.segmented)
            .disabled(viewModel.isWorking)

            if !viewModel.completionMessage.isEmpty {
                CompletionBanner(message: viewModel.completionMessage)
            }

            if viewModel.selectedMode == .upload {
                uploadSection
            } else {
                downloadSection
            }

            Spacer()
        }
    }

    private var connectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel("Připojení")
            HStack(spacing: 8) {
                SecureField("Skupinový API klíč", text: $viewModel.apiKey)
                    .textFieldStyle(.roundedBorder)
                    .disabled(viewModel.isWorking)
                Button {
                    viewModel.loadModels()
                } label: {
                    Label("Načíst modely", systemImage: "arrow.clockwise")
                }
                .disabled(viewModel.isWorking || viewModel.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            if !viewModel.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Button {
                    viewModel.forgetAPIKey()
                } label: {
                    Label("Zapomenout klíč", systemImage: "key.slash")
                }
                .buttonStyle(.link)
                .disabled(viewModel.isWorking)
            }
            Text(viewModel.modelStatus)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
    }

    private var uploadSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel("Nahrání")
            FolderPickerRow(
                title: "Složka se skeny",
                path: $viewModel.inputFolder,
                isDisabled: viewModel.isWorking,
                onPicked: viewModel.didPickInputFolder
            )
            TextField("Název titulu (volitelné)", text: $viewModel.titleName)
                .textFieldStyle(.roundedBorder)
                .disabled(viewModel.isWorking)
            ModelPickerRow(
                title: "Model ořezu",
                selection: $viewModel.cropModel,
                options: viewModel.cropModels,
                isDisabled: viewModel.isWorking
            )
            ModelPickerRow(
                title: "Model rotace",
                selection: $viewModel.rotationModel,
                options: viewModel.rotationModels,
                isDisabled: viewModel.isWorking
            )

            UploadReadinessPanel(viewModel: viewModel)

            HStack(spacing: 10) {
                Button {
                    viewModel.upload()
                } label: {
                    Label("Nahrát a počkat", systemImage: "arrow.up.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(viewModel.isWorking)

                Button {
                    viewModel.openEditor()
                } label: {
                    Label("Otevřít editor", systemImage: "safari")
                }
                .controlSize(.large)
                .disabled(viewModel.lastEditorURL == nil)

                Button {
                    viewModel.showDownload()
                } label: {
                    Label("Stáhnout", systemImage: "arrow.down.circle")
                }
                .controlSize(.large)
                .disabled(viewModel.generatedTitleID.isEmpty || viewModel.isWorking)
            }
            .padding(.top, 4)
        }
    }

    private var downloadSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel("Stažení")

            DownloadGuidancePanel(isGuidedReady: viewModel.canUseGuidedDownload)

            FolderPickerRow(
                title: "Kam uložit",
                path: $viewModel.outputFolder,
                isDisabled: viewModel.isWorking,
                onPicked: {}
            )

            DisclosureGroup(isExpanded: $viewModel.downloadsExistingBatch) {
                VStack(alignment: .leading, spacing: 12) {
                    FolderPickerRow(
                        title: "Původní složka",
                        path: $viewModel.existingBatchInputFolder,
                        isDisabled: viewModel.isWorking,
                        onPicked: {}
                    )
                    TextField("ID titulu", text: $viewModel.existingBatchTitleID)
                        .textFieldStyle(.roundedBorder)
                        .disabled(viewModel.isWorking)
                }
                .padding(.top, 8)
            } label: {
                Label("Stáhnout existující dávku", systemImage: "tray.and.arrow.down")
                    .font(.system(size: 13, weight: .medium))
            }
            .disabled(viewModel.isWorking)

            Button {
                viewModel.download()
            } label: {
                Label("Stáhnout a oříznout", systemImage: "arrow.down.circle.fill")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(viewModel.isWorking || (!viewModel.downloadsExistingBatch && !viewModel.canUseGuidedDownload))
            .padding(.top, 4)
        }
    }

    private var logColumn: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                SectionLabel("Průběh")
                Spacer()
                Button {
                    viewModel.clearLog()
                } label: {
                    Image(systemName: "trash")
                }
                .help("Vymazat záznam")
                .disabled(viewModel.logText.isEmpty)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(viewModel.logText.isEmpty ? "Připraveno." : viewModel.logText)
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(viewModel.logText.isEmpty ? .secondary : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(16)
                        .id("logEnd")
                }
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                )
                .onChange(of: viewModel.logText) { _ in
                    proxy.scrollTo("logEnd", anchor: .bottom)
                }
            }
        }
    }
}

private struct CompletionBanner: View {
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text(message)
                .font(.system(size: 13))
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.green.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct DownloadGuidancePanel: View {
    let isGuidedReady: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: isGuidedReady ? "checkmark.circle.fill" : "info.circle")
                .foregroundStyle(isGuidedReady ? .green : .secondary)
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var message: String {
        if isGuidedReady {
            return "Aplikace použije skeny a ID titulu z posledního nahrání. Vyberte jen složku, kam se uloží oříznuté obrázky."
        }
        return "Pro běžný postup nejdřív nahrajte skeny a uložte úpravy v editoru. Pokud už máte existující dávku, otevřete možnost níže."
    }
}

private struct UploadReadinessPanel: View {
    @ObservedObject var viewModel: CropilotViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: iconName)
                    .foregroundStyle(iconColor)
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
            }

            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)

            if !viewModel.generatedTitleID.isEmpty {
                HStack(spacing: 8) {
                    Text("ID titulu")
                        .foregroundStyle(.secondary)
                    Text(viewModel.generatedTitleID)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
                .font(.system(size: 12))
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var title: String {
        if viewModel.lastEditorURL != nil {
            return "Připraveno ke kontrole"
        }
        if viewModel.isWorking {
            return "Cropilot zpracovává skeny"
        }
        return "Editor se zobrazí po zpracování"
    }

    private var message: String {
        if viewModel.lastEditorURL != nil {
            return "Otevřete editor, zkontrolujte a uložte rámečky ořezu a potom stáhněte výsledky."
        }
        if viewModel.isWorking {
            return "Aplikace průběžně kontroluje stav. Toto okno můžete nechat otevřené."
        }
        return "Po nahrání aplikace počká, až Cropilot dokončí predikce, a teprve potom zobrazí odkaz do editoru."
    }

    private var iconName: String {
        if viewModel.lastEditorURL != nil {
            return "checkmark.circle.fill"
        }
        if viewModel.isWorking {
            return "clock.arrow.circlepath"
        }
        return "link.circle"
    }

    private var iconColor: Color {
        if viewModel.lastEditorURL != nil {
            return .green
        }
        if viewModel.isWorking {
            return .accentColor
        }
        return .secondary
    }
}

private struct SectionLabel: View {
    let title: String

    init(_ title: String) {
        self.title = title
    }

    var body: some View {
        Text(title)
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }
}

private struct FolderPickerRow: View {
    let title: String
    @Binding var path: String
    let isDisabled: Bool
    let onPicked: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            TextField(title, text: $path)
                .textFieldStyle(.roundedBorder)
                .disabled(isDisabled)
            Button {
                chooseFolder()
            } label: {
                Image(systemName: "folder")
            }
            .help("Vybrat složku")
            .disabled(isDisabled)
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Vybrat"
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
            onPicked()
        }
    }
}

private struct ModelPickerRow: View {
    let title: String
    @Binding var selection: String
    let options: [String]
    let isDisabled: Bool

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .frame(width: 96, alignment: .leading)
                .foregroundStyle(.secondary)
            Picker(title, selection: $selection) {
                Text("Výchozí").tag("")
                ForEach(options, id: \.self) { option in
                    Text(option).tag(option)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
            .disabled(isDisabled || options.isEmpty)
        }
    }
}

private struct StatusBadge: View {
    let state: WorkState

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(state.label)
                .font(.system(size: 13, weight: .medium))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var color: Color {
        switch state {
        case .ready:
            .secondary
        case .working:
            .accentColor
        case .done:
            .green
        case .failed:
            .red
        }
    }
}
