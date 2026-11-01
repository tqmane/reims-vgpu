#Requires -Version 7.2
. "$PSScriptRoot/../start.ps1" -FunctionsOnly
$testRoot=Join-Path $ProjectRoot ('.work/tmp/opencore-settings-'+[guid]::NewGuid())
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $scripts=Split-Path $PSScriptRoot
    foreach ($item in @(@('prepare-opencore.ps1','import json, pathlib, plistlib, re, sys, uuid','identity.py'),@('prepare-update-compatibility.ps1','import pathlib, plistlib, sys','updates.py'),@('prepare-opencore.ps1','import plistlib, pathlib, sys','picker.py'))) {
        $ast=[scriptblock]::Create((Get-Content -Raw (Join-Path $scripts $item[0]))).Ast
        $prefix=$item[1]
        $code=$ast.Find({param($n) $n -is [Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith($prefix)},$true).Value
        $code | Set-Content -LiteralPath (Join-Path $testRoot $item[2]) -Encoding utf8NoBOM
    }
    $test=@'
import json,pathlib,plistlib,sys,copy
r=pathlib.Path(sys.argv[1]); source=r/'original.plist'; dest=r/'candidate.plist'; identity=r/'identity.json'
guid='7C436110-AB2A-4BBB-A880-FE41995C9F82'
original={'PlatformInfo':{'Generic':{'SystemProductName':'iMac20,1','Other':'keep'}},
 'Kernel':{'Add':[{'BundlePath':'Keep.kext'}],'Patch':[{'Comment':'keep'}],'Quirks':{}},
 'NVRAM':{'Add':{guid:{'boot-args':'-v keepsyms=1','csr-active-config':b'\0'*4}}},
 'Misc':{'Security':{'SecureBootModel':'x86legacy'}}}
source.write_bytes(plistlib.dumps(original)); raw=source.read_bytes()
def run(name,*args):
 sys.argv=[name,*map(str,args)]; exec((r/name).read_text(encoding='utf-8-sig'),{})
valid=dict(model='iMac20,1',serial='TESTSERIAL12',mlb='TESTBOARD12345678',uuid='11111111-2222-3333-4444-555555555555',rom='525400112233')
identity.write_text(json.dumps(valid)); run('identity.py',source,dest,identity)
c=plistlib.loads(dest.read_bytes()); assert c['PlatformInfo']['Generic']['ROM']==bytes.fromhex(valid['rom'])
assert c['PlatformInfo']['Generic']['Other']=='keep' and c['Kernel']==original['Kernel']
assert source.read_bytes()==raw
for field,value in [('model','bad'),('serial','bad'),('mlb','bad'),('uuid','00000000-0000-0000-0000-000000000000'),('rom','015400112233'),('rom','000000000000')]:
 bad=dict(valid); bad[field]=value; identity.write_text(json.dumps(bad)); before=dest.read_bytes()
 try: run('identity.py',source,dest,identity)
 except (AssertionError,ValueError): pass
 else: raise AssertionError('Invalid identity accepted: '+field)
 assert dest.read_bytes()==before and source.read_bytes()==raw
run('updates.py',source,dest); c=plistlib.loads(dest.read_bytes())
assert [x['BundlePath'] for x in c['Kernel']['Add']]==['Lilu.kext','RestrictEvents.kext','Keep.kext']
assert c['Kernel']['Patch']==original['Kernel']['Patch'] and c['Misc']==original['Misc']
assert c['NVRAM']['Add'][guid]['csr-active-config']==b'\0'*4
assert c['NVRAM']['Add'][guid]['boot-args']=='-v keepsyms=1 revpatch=sbvmm revblock=none'
assert c['Kernel']['Quirks']['DisableLinkeditJettison']
source.write_bytes(dest.read_bytes()); run('updates.py',source,dest)
assert plistlib.loads(dest.read_bytes())==c
c['Misc']['Boot']={'Timeout':0,'ShowPicker':False,'Other':'keep'}
source.write_bytes(plistlib.dumps(c)); before=source.read_bytes()
run('picker.py',source,dest,5); chosen=plistlib.loads(dest.read_bytes())
expected=copy.deepcopy(c); expected['Misc']['Boot'].update(Timeout=5,ShowPicker=True)
expected['Misc']['Security']['AllowSetDefault']=True
assert chosen==expected and source.read_bytes()==before
for value in (-1,61):
 previous=dest.read_bytes()
 try: run('picker.py',source,dest,value)
 except AssertionError: pass
 else: raise AssertionError('Invalid picker timeout accepted')
 assert dest.read_bytes()==previous and source.read_bytes()==before
print('SMBIOS, update compatibility and boot picker checks passed.')
'@
    $test | & (Get-ReimsTool 'python.exe') - $testRoot
    if ($LASTEXITCODE -ne 0) { throw 'OpenCore settings checks failed.' }
} finally {
    $resolved=[IO.Path]::GetFullPath($testRoot)
    $allowed=[IO.Path]::GetFullPath((Join-Path $ProjectRoot '.work/tmp'))+[IO.Path]::DirectorySeparatorChar
    if ($resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
